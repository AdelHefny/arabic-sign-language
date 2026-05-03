export class MotionDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private lastImageData: ImageData | null = null;
  private motionScore: number = 0;

  constructor() {
    this.canvas = document.createElement("canvas");
    this.canvas.width = 64; // Low res for speed
    this.canvas.height = 48;
    this.ctx = this.canvas.getContext("2d")!;
  }

  calculateMotion(video: HTMLVideoElement): number {
    if (video.readyState < 2) return 0;
    
    this.ctx.drawImage(video, 0, 0, this.canvas.width, this.canvas.height);
    const currentImageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    
    if (!this.lastImageData) {
      this.lastImageData = currentImageData;
      return 0;
    }
    
    let diff = 0;
    for (let i = 0; i < currentImageData.data.length; i += 4) {
      diff += Math.abs(currentImageData.data[i] - this.lastImageData.data[i]);
    }
    
    this.lastImageData = currentImageData;
    this.motionScore = diff;
    return this.motionScore;
  }
}
