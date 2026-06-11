use std::sync::atomic::{AtomicU32, Ordering};

pub const JOYPAD_RIGHT: u32 = 0x01;
pub const JOYPAD_LEFT: u32 = 0x02;
pub const JOYPAD_UP: u32 = 0x04;
pub const JOYPAD_DOWN: u32 = 0x08;
pub const JOYPAD_A: u32 = 0x10;
pub const JOYPAD_B: u32 = 0x20;
pub const JOYPAD_SELECT: u32 = 0x40;
pub const JOYPAD_START: u32 = 0x80;

pub struct Joypad {
    state: AtomicU32,
}

impl Joypad {
    pub fn new() -> Self {
        Self { state: AtomicU32::new(0) }
    }

    pub fn press(&self, bits: u32) {
        self.state.fetch_or(bits, Ordering::Relaxed);
    }

    pub fn release(&self, bits: u32) {
        self.state.fetch_and(!bits, Ordering::Relaxed);
    }

    pub fn release_all(&self) {
        self.state.store(0, Ordering::Relaxed);
    }

    pub fn state(&self) -> u32 {
        self.state.load(Ordering::Relaxed)
    }

    pub fn is_idle(&self) -> bool {
        self.state() == 0
    }
}

pub fn key_to_bits(key: &str) -> Option<u32> {
    match key {
        "w" => Some(JOYPAD_UP),
        "s" => Some(JOYPAD_DOWN),
        "a" => Some(JOYPAD_LEFT),
        "d" => Some(JOYPAD_RIGHT),
        "k" => Some(JOYPAD_A),
        "j" => Some(JOYPAD_B),
        "h" => Some(JOYPAD_START),
        "g" => Some(JOYPAD_SELECT),
        _ => None,
    }
}

pub fn decode_joypad(bits: u32) -> (u8, u8) {
    let dpad = if bits & JOYPAD_RIGHT != 0 { 4 }
    else if bits & JOYPAD_LEFT != 0 { 3 }
    else if bits & JOYPAD_UP != 0 { 1 }
    else if bits & JOYPAD_DOWN != 0 { 2 }
    else { 0 };
    let btn = if bits & JOYPAD_A != 0 { 1 }
    else if bits & JOYPAD_B != 0 { 2 }
    else if bits & JOYPAD_START != 0 { 3 }
    else if bits & JOYPAD_SELECT != 0 { 4 }
    else { 0 };
    (dpad, btn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_joypad_new_is_idle() {
        let j = Joypad::new();
        assert!(j.is_idle());
        assert_eq!(j.state(), 0);
    }

    #[test]
    fn test_joypad_press_and_release() {
        let j = Joypad::new();
        j.press(JOYPAD_A);
        assert_eq!(j.state(), JOYPAD_A);
        j.release(JOYPAD_A);
        assert!(j.is_idle());
    }

    #[test]
    fn test_joypad_multi_press() {
        let j = Joypad::new();
        j.press(JOYPAD_UP | JOYPAD_A);
        assert!(j.state() & JOYPAD_UP != 0);
        assert!(j.state() & JOYPAD_A != 0);
    }

    #[test]
    fn test_joypad_release_one() {
        let j = Joypad::new();
        j.press(JOYPAD_UP | JOYPAD_A | JOYPAD_B);
        j.release(JOYPAD_A);
        let s = j.state();
        assert!(s & JOYPAD_UP != 0);
        assert!(s & JOYPAD_B != 0);
        assert_eq!(s & JOYPAD_A, 0);
    }

    #[test]
    fn test_joypad_release_all() {
        let j = Joypad::new();
        j.press(JOYPAD_UP | JOYPAD_DOWN | JOYPAD_A);
        j.release_all();
        assert!(j.is_idle());
    }

    #[test]
    fn test_key_to_bits() {
        assert_eq!(key_to_bits("w"), Some(JOYPAD_UP));
        assert_eq!(key_to_bits("x"), None);
        assert_eq!(key_to_bits(""), None);
    }

    #[test]
    fn test_decode_joypad_dpad() {
        assert_eq!(decode_joypad(JOYPAD_RIGHT), (4, 0));
        assert_eq!(decode_joypad(JOYPAD_LEFT), (3, 0));
        assert_eq!(decode_joypad(JOYPAD_UP), (1, 0));
        assert_eq!(decode_joypad(JOYPAD_DOWN), (2, 0));
    }

    #[test]
    fn test_decode_joypad_btn() {
        assert_eq!(decode_joypad(JOYPAD_A), (0, 1));
        assert_eq!(decode_joypad(JOYPAD_B), (0, 2));
        assert_eq!(decode_joypad(JOYPAD_START), (0, 3));
        assert_eq!(decode_joypad(JOYPAD_SELECT), (0, 4));
    }

    #[test]
    fn test_double_press_idempotent() {
        let j = Joypad::new();
        j.press(JOYPAD_A);
        j.press(JOYPAD_A);
        assert_eq!(j.state(), JOYPAD_A);
        j.release(JOYPAD_A);
        assert!(j.is_idle());
    }
}
