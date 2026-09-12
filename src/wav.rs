//! Minimal RIFF/WAVE reader for 16-bit PCM, used by tools and tests.

use crate::kaldi_io::err;
use std::io::Result;

pub struct Wav {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<i16>,
}

pub fn read_wav(path: &std::path::Path) -> Result<Wav> {
    let b = std::fs::read(path)?;
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(err("not a RIFF/WAVE file"));
    }
    let mut p = 12;
    let mut sample_rate = 0;
    let mut channels = 0;
    let mut bits = 0;
    let mut data: Option<&[u8]> = None;
    while p + 8 <= b.len() {
        let id = &b[p..p + 4];
        let size = u32::from_le_bytes(b[p + 4..p + 8].try_into().unwrap()) as usize;
        let body = &b[p + 8..(p + 8 + size).min(b.len())];
        if id == b"fmt " && body.len() >= 16 {
            channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
        } else if id == b"data" {
            data = Some(body);
        }
        p += 8 + size + (size & 1);
    }
    let data = data.ok_or_else(|| err("no data chunk"))?;
    if bits != 16 {
        return Err(err("only 16-bit PCM is supported"));
    }
    let samples = data
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    Ok(Wav {
        sample_rate,
        channels,
        samples,
    })
}
