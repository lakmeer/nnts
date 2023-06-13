
import { rankColor } from "./utils.ts";


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

const doFrame = (args = []) => {
  frameFn(ctx, size, args);
}

const setAspect = (aspect) => {
  aspect = aspect;
}

export const clear = (x = 0, y = 0, w = size.w, h = size.h) => {
  ctx.fillStyle = BLACK;
  ctx.fillRect(x, y, w, h);
}

export const line = (c, x1, y1, x2, y2, w = 1) => {
  ctx.lineWidth = 2 * w;
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

export const pad = (size, scale, fn) => {
  zone(size.w*(1-scale)/2, size.h*(1-scale)/2, size.w*scale, size.h*scale, fn);
}

export const all = (fn) => zone(0, 0, size.w, size.h, fn);

export const text = (text, x, y, c, size) => {
  ctx.fillStyle = c;
  ctx.font = `${size}px monospace`;
  ctx.fillText(text, x, y);
}

export const image = (img, size, x = 0, y = 0) => {
  ctx.drawImage(img, x, y, size.w, size.h);
}


//
// Specialised Drawing API
//

import type { NN } from './nn';
import * as Mat from './matrix';
import { min, rgb, weightColor, plainColor, log10 } from './utils';


// Draw a whole network
// - radiusScale makes the visual elements bigger
// - weightScale bumps heterogeneous weight values into a visible color range.

export const drawNetwork = (net:Net, { w, h }, radiusScale = 1, weightScale = 1, noColor = false) => {
  const numLayers = net.arch.length;
  const xStride = w/net.arch.length;

  const r = min(h/35, radiusScale * 0.9 * h / (2 * Math.max(...net.arch)));

  const color = noColor ? plainColor : weightColor;

  // Each layer
  for (let layer = 0; layer < numLayers; layer++) {
    const numNeurons = net.arch[layer];
    const yStride = h/numNeurons;
    const x = xStride/2 + layer * xStride;
    const c = `hsl(${ 360/numLayers * layer }, 100%, 50%)`;

    // Draw connections to next layer
    for (let i = 0; i < numNeurons; i++) {
      const y = yStride/2 + i * yStride;

      if (layer < numLayers - 1) {
        const nextNumNeurons = net.arch[layer+1];
        const nextYStride = h/nextNumNeurons;

        for (let j = 0; j < nextNumNeurons; j++) {
          const w = Mat.at(net.ws[layer], i, j);
          const nextY = nextYStride/2 + j * nextYStride;
          line(rgb(color(w, weightScale)), x, y, x+xStride, nextY, radiusScale*radiusScale);
        }
      }
    }

    // Draw neurons
    for (let i = 0; i < numNeurons; i++) {
      const y = yStride/2 + i * yStride;

      const b = layer == 0 ? 0 : Mat.at(net.bs[layer-1], 0, i);
      circle(rgb(color(b, weightScale)), x, y, r * radiusScale);
    }
  }
}


// Time Series Plot
// - expectedY - maximum expected value of data points in the series

export const plotSeries = (data, { w, h }, color, fill = false) => {
  const xStep = w/data.length;
  const maxY = Math.max(...data);

  for (let r = 0; r < 6; r++) {
    const t = Math.pow(10, -r);
    const lineY = h - t/maxY * h;
    ctx.beginPath();
    ctx.setLineDash([15, 15]);
    ctx.strokeStyle = rankColor(t / 10); 
    ctx.moveTo(0, lineY);
    ctx.lineTo(w, lineY);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.beginPath();
  ctx.moveTo(0, h);
  ctx.lineTo(0, h - h * data[0]/maxY);

  for (let i = 0; i < data.length - 1; i++) {
    const x1 = i * xStep;
    const y1 = h - h * data[i]/maxY;
    ctx.lineTo(x1, y1);

    if (i == data.length - 2) {
      ctx.lineTo(w, y1);
    }
  }

  if (fill) {
    ctx.globalAlpha = 0.1;
    ctx.fillStyle = rankColor(data[data.length-1]);
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.fill();
  }

  ctx.globalAlpha = 1.0;
  ctx.strokeStyle = rankColor(data[data.length-1]);
  ctx.lineWidth = 2;
  ctx.stroke();

  line('white', 0, h, w, h, 1);
}

export const fillSeries = (data, { w, h }, color) => {
  const xStep = w/data.length;
  const maxY =  Math.max(...data);

  ctx.beginPath();

  for (let i = 0; i < data.length - 1; i++) {
    const x1 = i * xStep;
    const y1 = h - h * data[i]/maxY;
    //const x2 = (i+1) * xStep;
    //const y2 = h - h * data[i+1]/maxY;
    ctx.lineTo(x1, y1);
  }

}




// Listeners

window.addEventListener('resize', setCanvasSize);


// Init

setupCanvas();


// Interface

const onFrame = (fn) => {
  frameFn = fn;
  requestAnimationFrame(doFrame.bind(null, []));
}

const poke = (...args) => {
  requestAnimationFrame(doFrame.bind(null, args));
}

export {
  canvas,
  ctx,
  size,
  onFrame,
  poke,
  setAspect,
}

