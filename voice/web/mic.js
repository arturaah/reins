class MicrophoneCapture extends AudioWorkletProcessor {
  constructor() {
    super(); this.frames = new Int16Array(2048); this.offset = 0; this.active = true;
    this.port.onmessage = ({data}) => {
      if (data === 'flush') {
        if (this.offset) this.port.postMessage(this.frames.slice(0, this.offset).buffer);
        this.active = false; this.port.postMessage('flushed');
      }
    };
  }
  process(inputs) {
    if (!this.active) return true;
    for (const sample of inputs[0]?.[0] || []) {
      const s = Math.max(-1, Math.min(1, sample));
      this.frames[this.offset++] = s < 0 ? s * 32768 : s * 32767;
      if (this.offset === this.frames.length) {
        this.port.postMessage(this.frames.buffer, [this.frames.buffer]);
        this.frames = new Int16Array(2048); this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor('microphone-capture', MicrophoneCapture);
