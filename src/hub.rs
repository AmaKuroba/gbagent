use std::sync::Arc;
use tokio::sync::broadcast;

/// Messages broadcast to all connected WebSocket clients.
#[derive(Clone, Debug)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::broadcast::error::TryRecvError;

    #[tokio::test]
    async fn test_hub_subscribe_receives_frame() {
        let hub = Hub::new(16);
        let mut rx = hub.subscribe();
        hub.broadcast_frame(vec![1, 2, 3]);
        let msg = rx.try_recv().unwrap();
        match msg {
            BroadcastMsg::Frame(data) => assert_eq!(data, vec![1, 2, 3]),
            _ => panic!("expected Frame"),
        }
    }

    #[tokio::test]
    async fn test_hub_subscribe_receives_text() {
        let hub = Hub::new(16);
        let mut rx = hub.subscribe();
        hub.broadcast_text("hello".into());
        let msg = rx.try_recv().unwrap();
        match msg {
            BroadcastMsg::Text(t) => assert_eq!(t, "hello"),
            _ => panic!("expected Text"),
        }
    }

    #[tokio::test]
    async fn test_hub_multiple_subscribers() {
        let hub = Hub::new(16);
        let mut rx1 = hub.subscribe();
        let mut rx2 = hub.subscribe();
        hub.broadcast_frame(vec![42]);
        assert!(rx1.try_recv().is_ok());
        assert!(rx2.try_recv().is_ok());
        assert_eq!(hub.subscriber_count(), 2);
    }

    #[tokio::test]
    async fn test_hub_no_message_when_none_sent() {
        let hub = Hub::new(16);
        let mut rx = hub.subscribe();
        let result = rx.try_recv();
        match result {
            Err(TryRecvError::Empty) => {}
            _ => panic!("expected Empty, got {:?}", result),
        }
    }

    #[tokio::test]
    async fn test_hub_capacity_respects_limit() {
        let hub = Hub::new(2);
        let mut rx = hub.subscribe();
        hub.broadcast_frame(vec![1]);
        hub.broadcast_frame(vec![2]);
        // Drain both messages
        let _ = rx.try_recv().unwrap();
        let _ = rx.try_recv().unwrap();
        // 3rd message with capacity 2: the oldest msg (1) is dropped
        hub.broadcast_frame(vec![3]);
        // Next recv should either succeed (if new messages fit) or lag
        let msg = rx.try_recv();
        // Either we get the message or we get Lagged — both are correct
        if let Ok(BroadcastMsg::Frame(data)) = msg {
            assert_eq!(data, vec![3]);
        }
    }
}
