class LiveMicrophone extends AudioWorkletProcessor {
  constructor() { super(); this.frames=new Int16Array(320); this.offset=0; }
  process(inputs) {
    for(const sample of inputs[0]?.[0] || []) {
      const value=Math.max(-1,Math.min(1,sample));
      this.frames[this.offset++]=value<0 ? value*32768 : value*32767;
      if(this.offset===320) {
        this.port.postMessage(this.frames.buffer,[this.frames.buffer]);
        this.frames=new Int16Array(320); this.offset=0;
      }
    }
    return true;
  }
}
registerProcessor('live-microphone',LiveMicrophone);
