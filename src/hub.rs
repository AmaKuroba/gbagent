use std::sync::Arc;
use tokio::sync::broadcast;

/// Messages broadcast to all connected WebSocket clients.
#[derive(Clone)]
pub enum BroadcastMsg {
    /// Binary frame data (PNG image).
    Frame(Vec<u8>),
    /// Text metrics data (JSON).
    Text(String),
}

/// Hub manages fan-out of broadcast messages to connected WebSocket clients.
pub struct Hub {
    tx: broadcast::Sender<BroadcastMsg>,
}

impl Hub {
    /// Create a new Hub with the given channel capacity.
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Subscribe to receive broadcast messages.
    pub fn subscribe(&self) -> broadcast::Receiver<BroadcastMsg> {
        self.tx.subscribe()
    }

    /// Broadcast a frame to all connected clients.
    pub fn broadcast_frame(&self, data: Vec<u8>) {
        let _ = self.tx.send(BroadcastMsg::Frame(data));
    }

    /// Broadcast text to all connected clients.
    pub fn broadcast_text(&self, text: String) {
        let _ = self.tx.send(BroadcastMsg::Text(text));
    }

    /// Number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }

    /// Create a shared reference.
    pub fn shared(self) -> Arc<Hub> {
        Arc::new(self)
    }
}
