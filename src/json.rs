//! The JSON the runtime reads and writes: a parser for an array of strings (the grammar) and a
//! writer with string escaping for the result shapes.
// [[rr:TD-2#Dependency policy]]

use crate::kaldi_io::err;
use std::io::Result;

/// Parse a JSON array of strings; anything else is an error.
pub fn parse_string_array(text: &str) -> Result<Vec<String>> {
    let b = text.as_bytes();
    let mut p = 0;
    let skip_ws = |p: &mut usize| {
        while *p < b.len() && (b[*p] as char).is_whitespace() {
            *p += 1;
        }
    };
    skip_ws(&mut p);
    if p >= b.len() || b[p] != b'[' {
        return Err(err("grammar must be a JSON array of strings"));
    }
    p += 1;
    let mut out = Vec::new();
    loop {
        skip_ws(&mut p);
        if p >= b.len() {
            return Err(err("unterminated grammar array"));
        }
        if b[p] == b']' {
            return Ok(out);
        }
        if b[p] != b'"' {
            return Err(err("grammar entries must be strings"));
        }
        p += 1;
        let mut s = String::new();
        loop {
            if p >= b.len() {
                return Err(err("unterminated string"));
            }
            let c = b[p];
            p += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    if p >= b.len() {
                        return Err(err("bad escape"));
                    }
                    let e = b[p];
                    p += 1;
                    match e {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'b' => s.push('\u{8}'),
                        b'f' => s.push('\u{c}'),
                        b'n' => s.push('\n'),
                        b'r' => s.push('\r'),
                        b't' => s.push('\t'),
                        b'u' => {
                            if p + 4 > b.len() {
                                return Err(err("bad unicode escape"));
                            }
                            let hex = std::str::from_utf8(&b[p..p + 4]).map_err(|_| err("bad unicode escape"))?;
                            let mut code = u32::from_str_radix(hex, 16).map_err(|_| err("bad unicode escape"))?;
                            p += 4;
                            if (0xD800..0xDC00).contains(&code) && p + 6 <= b.len() && &b[p..p + 2] == b"\\u" {
                                let hex2 = std::str::from_utf8(&b[p + 2..p + 6]).map_err(|_| err("bad unicode escape"))?;
                                let low = u32::from_str_radix(hex2, 16).map_err(|_| err("bad unicode escape"))?;
                                if (0xDC00..0xE000).contains(&low) {
                                    code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                                    p += 6;
                                }
                            }
                            s.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                        }
                        _ => return Err(err("bad escape")),
                    }
                }
                _ => {
                    // Copy one UTF-8 sequence.
                    let start = p - 1;
                    let len = match c {
                        0x00..=0x7F => 1,
                        0xC0..=0xDF => 2,
                        0xE0..=0xEF => 3,
                        _ => 4,
                    };
                    let end = (start + len).min(b.len());
                    s.push_str(std::str::from_utf8(&b[start..end]).map_err(|_| err("invalid UTF-8"))?);
                    p = end;
                }
            }
        }
        out.push(s);
        skip_ws(&mut p);
        if p < b.len() && b[p] == b',' {
            p += 1;
        } else if p < b.len() && b[p] == b']' {
            return Ok(out);
        } else {
            return Err(err("expected , or ] in grammar array"));
        }
    }
}

pub fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_strings_with_escapes() {
        let v = parse_string_array(r#" [ "king", "a b", "x\"y", "é" ] "#).unwrap();
        assert_eq!(v, vec!["king", "a b", "x\"y", "é"]);
        assert!(parse_string_array("[1]").is_err());
        assert_eq!(parse_string_array("[]").unwrap().len(), 0);
    }
}
