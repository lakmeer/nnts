
//
// Index file is just for setting up the canvas and providing sensible renderign APIs.
//

const canvas = document.getElementById("canvas");
const ctx    = canvas.getContext("2d");


// Config

const BLACK = "#212121";
const WHITE = "#fffefc";


// State

let size = { w: 0, h: 0 };
let aspect = 1;

let frameFn = () => {};


// Functions

const setupCanvas = () => {
  canvas.style.position = "absolute";
  canvas.style.top = "0px";
  canvas.style.left = "0px";
  document.body.style.backgroundColor = BLACK;
  setCanvasSize();
  resetCanvas();
}

const setCanvasSize = () => {
  size.w = window.innerWidth;
  size.h = window.innerHeight;

  if (size.w/size.h > aspect) {
    size.w = size.h/aspect;
  } else {
    size.h = size.w/aspect;
  }

  canvas.width  = size.w;
  canvas.height = size.h;

  resetCanvas();
  doFrame();
}

const resetCanvas = () => {
  ctx.fillStyle = WHITE;
}


// Simplified drawing API

const doFrame = () => {
  frameFn(ctx, size);
}

const setAspect = (aspect) => {
  aspect = aspect;
}

export const line = (c, x1, y1, x2, y2) => {
  ctx.strokeStyle = c;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

export const circle = (c, x, y, r) => {
  ctx.fillStyle = c;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fill();
}

export const grid = (c, x, y, w, h, s) => {
  ctx.strokeStyle = c;
  ctx.beginPath();
  for (let i = x; i < x+w+1; i += w/s) {
    ctx.moveTo(i, y);
    ctx.lineTo(i, y+h);
  }
  for (let i = y; i < y+h+1; i += h/s) {
    ctx.moveTo(x, i);
    ctx.lineTo(x+w, i);
  }
  ctx.stroke();
}

export const zone = (x, y, w, h, fn) => {
  ctx.save();
  ctx.translate(x, y);
  fn(ctx, { w, h });
  ctx.restore();
}


// Listeners

window.addEventListener('resize', setCanvasSize);


// Init

setupCanvas();


// Interface

const onFrame = (fn) => {
  frameFn = fn;
  requestAnimationFrame(doFrame);
}

export {
  canvas,
  ctx,
  size,
  onFrame,
  setAspect,
}

