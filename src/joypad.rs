use std::sync::atomic::{AtomicU32, Ordering};

/// Joypad button bit constants (active-high internal state).
pub const JOYPAD_RIGHT: u32 = 0x01;
pub const JOYPAD_LEFT: u32 = 0x02;
pub const JOYPAD_UP: u32 = 0x04;
pub const JOYPAD_DOWN: u32 = 0x08;
pub const JOYPAD_A: u32 = 0x10;
pub const JOYPAD_B: u32 = 0x20;
pub const JOYPAD_SELECT: u32 = 0x40;
pub const JOYPAD_START: u32 = 0x80;

/// Thread-safe joypad state using atomic operations.
/// Bit layout: R(0) L(1) U(2) D(3) A(4) B(5) Select(6) Start(7).
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

/// Map keyboard key names (from the dashboard) to joypad bitmasks.
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

/// Extract dpad index (0-4) and btn index (0-4) from a joypad state value.
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
