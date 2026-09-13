// Vendored from Vosk-Rust (Apache-2.0), see third_party/Vosk-Rust/NOTICE.
//! Reader for Kaldi's binary serialization (the `\0B` format).
//!
//! Tokens (`<Foo>`) are space-terminated ASCII even in binary mode. Basic types are written as
//! `[size-byte][little-endian bytes]` (int32/float: size byte 4, double: size byte 8). Integer
//! vectors are `[i32 count][raw i32 x count]`. Float matrices are token `FM` then
//! `[i32 rows][i32 cols]` then `rows*cols` raw f32; float vectors are token `FV` then `[i32 dim]`
//! then `dim` raw f32; `DM`/`DV` are the double forms, `DP` a packed symmetric double matrix.

use std::io::{Error, ErrorKind, Read, Result};

pub struct KaldiReader<R: Read> {
    r: R,
}

fn cells(rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| err("matrix dimensions overflow the address space"))
}

pub fn err(msg: &str) -> Error {
    Error::new(ErrorKind::InvalidData, msg.to_string())
}

impl<R: Read> KaldiReader<R> {
    pub fn new(r: R) -> Self {
        KaldiReader { r }
    }

    #[inline]
    fn byte(&mut self) -> Result<u8> {
        let mut b = [0u8; 1];
        self.r.read_exact(&mut b)?;
        Ok(b[0])
    }

    pub fn expect_binary(&mut self) -> Result<()> {
        let a = self.byte()?;
        let b = self.byte()?;
        if a != 0 || b != b'B' {
            return Err(err("not a binary Kaldi stream (missing \\0B)"));
        }
        Ok(())
    }

    /// Read one space-terminated token (skips leading whitespace).
    pub fn read_token(&mut self) -> Result<String> {
        let mut s = String::new();
        loop {
            let c = self.byte()?;
            if c == b' ' || c == b'\n' || c == b'\t' {
                if s.is_empty() {
                    continue;
                }
                break;
            }
            s.push(c as char);
        }
        Ok(s)
    }

    pub fn expect_token(&mut self, want: &str) -> Result<()> {
        let got = self.read_token()?;
        if got != want {
            return Err(err(&format!("expected {want}, got {got}")));
        }
        Ok(())
    }

    pub fn read_i32(&mut self) -> Result<i32> {
        let sz = self.byte()?;
        if sz != 4 {
            return Err(err("expected i32 size byte = 4"));
        }
        let mut b = [0u8; 4];
        self.r.read_exact(&mut b)?;
        Ok(i32::from_le_bytes(b))
    }

    /// An i32 count or dimension: a negative one would wrap into an enormous allocation.
    pub fn read_dim(&mut self) -> Result<usize> {
        let n = self.read_i32()?;
        usize::try_from(n).map_err(|_| err(&format!("negative dimension {n}")))
    }

    pub fn read_f32(&mut self) -> Result<f32> {
        let sz = self.byte()?;
        if sz != 4 {
            return Err(err("expected f32 size byte = 4"));
        }
        let mut b = [0u8; 4];
        self.r.read_exact(&mut b)?;
        Ok(f32::from_le_bytes(b))
    }

    pub fn read_f64(&mut self) -> Result<f64> {
        let sz = self.byte()?;
        if sz != 8 {
            return Err(err("expected f64 size byte = 8"));
        }
        let mut b = [0u8; 8];
        self.r.read_exact(&mut b)?;
        Ok(f64::from_le_bytes(b))
    }

    /// Kaldi writes a bool as the single character `T` or `F`.
    pub fn read_bool(&mut self) -> Result<bool> {
        match self.byte()? {
            b'T' => Ok(true),
            b'F' => Ok(false),
            c => Err(err(&format!("expected T or F for a bool, got {c}"))),
        }
    }

    /// `count` elements of `width` bytes, grown as the bytes arrive rather than sized from the
    /// count: a length field alone then buys no allocation the stream cannot fill.
    fn read_bounded(&mut self, count: usize, width: usize) -> Result<Vec<u8>> {
        let want = count
            .checked_mul(width)
            .ok_or_else(|| err("length overflows the address space"))?;
        let mut buf = Vec::new();
        let got = self.r.by_ref().take(want as u64).read_to_end(&mut buf)?;
        if got != want {
            return Err(err("unexpected end of Kaldi stream"));
        }
        Ok(buf)
    }

    pub fn read_i32_vec(&mut self) -> Result<Vec<i32>> {
        let n = self.read_dim()?;
        let buf = self.read_bounded(n, 4)?;
        Ok(buf
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect())
    }

    fn read_f32_raw(&mut self, count: usize) -> Result<Vec<f32>> {
        let buf = self.read_bounded(count, 4)?;
        Ok(buf
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    fn read_f64_raw(&mut self, count: usize) -> Result<Vec<f64>> {
        let buf = self.read_bounded(count, 8)?;
        Ok(buf
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect())
    }

    pub fn read_float_vec(&mut self) -> Result<Vec<f32>> {
        self.expect_token("FV")?;
        let dim = self.read_dim()?;
        self.read_f32_raw(dim)
    }

    /// Returns (rows, cols, row-major data).
    pub fn read_float_matrix(&mut self) -> Result<(usize, usize, Vec<f32>)> {
        self.expect_token("FM")?;
        let rows = self.read_dim()?;
        let cols = self.read_dim()?;
        let data = self.read_f32_raw(cells(rows, cols)?)?;
        Ok((rows, cols, data))
    }

    pub fn read_double_vec(&mut self) -> Result<Vec<f64>> {
        self.expect_token("DV")?;
        let dim = self.read_dim()?;
        self.read_f64_raw(dim)
    }

    pub fn read_double_matrix(&mut self) -> Result<(usize, usize, Vec<f64>)> {
        self.expect_token("DM")?;
        let rows = self.read_dim()?;
        let cols = self.read_dim()?;
        let data = self.read_f64_raw(cells(rows, cols)?)?;
        Ok((rows, cols, data))
    }

    /// Packed symmetric double matrix: `DP`, dim, then dim*(dim+1)/2 values, lower triangle
    /// row by row (Kaldi's `SpMatrix` layout). Returns (dim, packed data).
    pub fn read_packed_double(&mut self) -> Result<(usize, Vec<f64>)> {
        self.expect_token("DP")?;
        let dim = self.read_dim()?;
        let data = self.read_f64_raw(cells(dim, dim + 1)? / 2)?;
        Ok((dim, data))
    }
}

/// Kaldi text-form matrix (` [ a b c\n d e f ]`), the form `global_cmvn.stats` ships in.
pub fn parse_text_matrix(txt: &str) -> Result<(usize, usize, Vec<f64>)> {
    let body = txt.trim();
    let body = body
        .strip_prefix('[')
        .ok_or_else(|| err("text matrix must start with ["))?;
    let body = body
        .strip_suffix(']')
        .ok_or_else(|| err("text matrix must end with ]"))?;
    let mut rows = 0;
    let mut cols = 0;
    let mut data = Vec::new();
    for line in body.lines() {
        let vals: Vec<f64> = line
            .split_whitespace()
            .map(|v| {
                v.parse::<f64>()
                    .map_err(|_| err(&format!("bad number {v}")))
            })
            .collect::<Result<_>>()?;
        if vals.is_empty() {
            continue;
        }
        if cols == 0 {
            cols = vals.len();
        } else if cols != vals.len() {
            return Err(err("ragged text matrix"));
        }
        rows += 1;
        data.extend(vals);
    }
    Ok((rows, cols, data))
}

pub fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from > hay.len() {
        return None;
    }
    hay[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| p + from)
}
