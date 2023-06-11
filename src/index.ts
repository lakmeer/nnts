
//
// NN.ts
//
// Reimplemented from Tsoding (https://twitch.tv/tsoding)
// by watching recordings (https://youtube.com/@TsodingDaily),
// not by copying the code (https://github.com/tsoding/nn.h).
//
// This is just to scratch my own itch, do not use lol.
//

import { weightColor, logHelper, limit, costRank, rgb, abs, max } from './utils';

import * as Screen from './canvas';
import * as Mat from './matrix';
import * as NN from './nn';

type TrainState =
  | "IDLE"
  | "TRAINING"
  | "CANCELLED"
  | "STALLED"
  | "STOPPED"
  | "FINISHED";

const log = logHelper("IMG");


//
// Training Loop
//

const PREVIEW_MODE: "OUTPUT" | "DIFF" | "DEBUG" = "OUTPUT";

export const train = async (net:NN, ti, to, options = {}, img, w, h, upSize = 2) => {

  const maxSteps   = options.maxSteps  ?? 10000;
  const maxRank    = options.maxRank   ?? 4;
  const rate       = options.rate      ?? 1;
  const epochSize  = options.epochSize ?? 100;
  const aggression = options.aggression ?? 0;
  const jitter     = options.jitter    ?? 0;

  const maxWindow = 100;

  Screen.setAspect(1);


  const cancel = () => state = "CANCELLED";
  document.addEventListener('keydown', cancel);


  // Prepare Gradient Network

  const grad = NN.alloc(net.arch);

  log.info(`Training for ${maxSteps} steps...`);

  let c = 1;
  let rank = 0;
  let costHist = [];
  let step = 0;
  let state: TrainState = "TRAINING";

  const input   = newSurface(w, h, img);
  const diff    = newSurface(w, h);
  const output  = newSurface(w, h, img);
  const upscale = newSurface(w * upSize, h * upSize);


  // Training loop

  const trainBatch = async () => {

    // Scale learning rate as we progress
    const a = limit(1, 10, 1/(c + 1 - aggression * (step/maxSteps)));
    const r = rate/2 + rate/2 * a;

    // Apply backpropagated gradient
    for (let i = 0; i < epochSize; i++) {

      NN.backprop(net, grad, 0, ti, to);
      NN.learn(net, grad, r);
      c = NN.cost(net, ti, to);
      costHist.push(c);
    }

    // Review batch
    step += epochSize;
    rank = costRank(c);

    if (step >= maxSteps) state = "STOPPED";
    if (rank >= maxRank)  state = "FINISHED";


    // Run inference and paint to canvas(es)

    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        Mat.put(net.as[0], 0, 0, x/w);
        Mat.put(net.as[0], 0, 1, y/h);

        NN.forward(net);

        const i = (x + y * h);
        const b = abs(Mat.at(net.as[net.count], 0, 0));
        const expect = input.pget(x, y)[0]/255;

        output.vset(x, y, [ b, b, b ]);
        diff.pset(x, y, weightColor(expect - b, 1, true));
      }
    }

    // Commit pixels to surface
    output.update();
    diff.update();

    // Render net
    netToImg(net, output, w, h);

    // Only render upscale of factor n every nth epoch
    if (step % (epochSize * upSize) === 0) {
      netToImg(net, upscale, w, h, upSize);
    }

    // Draw new frame
    Screen.all((ctx, { w, h }) => {
      Screen.clear();

      Screen.grid('grey', 0, 0, w, h, 8);

      // Network diagram
      Screen.zone(0, h/4, w*3/4, h*3/4, (ctx, size) => {
        Screen.pad(size, 0.9, (ctx, size) => {
          Screen.drawNetwork(grad, size, 1.2, 10, true);
          Screen.drawNetwork(net,  size, 1,   1);
        });
      });

      // Input image
      Screen.zone(w*3/4, h*0/4, w/4, h/4, (ctx, size) => {
        Screen.pad(size, 0.9, (ctx, size) => {
          Screen.image(input.canvas, size);
        });
      });

      // Diff image
      Screen.zone(w*3/4, h*1/4, w/4, h/4, (ctx, size) => {
        Screen.pad(size, 0.9, (ctx, size) => {
          Screen.image(diff.canvas, size);
        });
      });

      // Output image
      Screen.zone(w*3/4, h*2/4, w/4, h/4, (ctx, size) => {
        Screen.pad(size, 0.9, (ctx, size) => {
          Screen.image(output.canvas, size);
        });
      });

      // Upscale preview
      Screen.zone(w*3/4, h*3/4, w/4, h/4, (ctx, size) => {
        Screen.pad(size, 0.9, (ctx, size) => {
          Screen.image(upscale.canvas, size);
        });
      });

      // Cost history plot
      Screen.zone(0, 0, w*3/4, h/4, (ctx, size) => {
        Screen.plotSeries(costHist, size, options.color ?? '#ff2266', true);

        const tCost = c.toFixed(maxRank - 1);
        const tRate = r.toFixed(2);
        const tTime = ((performance.now() - start)/1000).toFixed(2);
        const tRank = costRank(c, true);

        Screen.text(`Cost: ${tCost}  Rate: ${tRate}  Epoch: ${step}/${maxSteps}  ${tTime}s`, 10, 25, 'white', 22);
        Screen.text(`${state} ${tRank}`, w - 160, 25, 'white', 22);
        Screen.text(`Set size: ${ti.rows}`, 10, size.h - 10, 'white', 22);
      });
    });

    // Stop if we're not improving
    if (state === "TRAINING") {
      await new Promise(requestAnimationFrame);
      await trainBatch();
    }

    // Gym might be interested in the final gradient network
    return grad;
  }


  // Begin training

  const start = performance.now();
  await trainBatch();
  const time = performance.now() - start;

  if (rank < maxRank) {
    log.err(`Stopping at rank ${rank} after ${step} steps.`);
  } else {
    log.ok(`Finished in ${time.toFixed(2)}ms and ${step} steps`);
  }

  document.removeEventListener('keydown', cancel);
}



//
// Helpers
//

const loadImage = (src:string) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  })

const imageToMatrix = (img:HTMLImageElement) => {
  const canvas = document.createElement('canvas');
  canvas.width  = img.width;
  canvas.height = img.height;
  canvas.style.top = 0;
  canvas.style.zIndex = '1000';
  canvas.style.position = 'absolute';

  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);

  const data = ctx.getImageData(0, 0, img.width, img.height).data;
  const mat  = Mat.alloc(img.width - 1, img.height - 1);

  const perceptualWeights = [0.2126, 0.7152, 0.0722];

  for (let x = 0; x < img.width - 1; x++) {
    for (let y = 0; y < img.height - 1; y++) {
      const i = (x + y * img.width) * 4;
      const r = data[i + 0] * perceptualWeights[0];
      const g = data[i + 1] * perceptualWeights[1];
      const b = data[i + 2] * perceptualWeights[2];
      const v = (r + g + b) / 255;
      Mat.put(mat, y, x, v);
    }
  }


  return mat;
}

const netToImg = (net:Network, surface:Surface, w:number, h:number, z = 1) => {
  w = w * z;
  h = h * z;

  for (let x = 0; x < w; x += 1) {
    for (let y = 0; y < h; y += 1) {
      Mat.put(net.as[0], 0, 0, x/w);
      Mat.put(net.as[0], 0, 1, y/h);

      NN.forward(net);

      const b = Mat.get(net.as[net.count], 0, 0);
      surface.data[(x + y * w) * 4 + 0] = b * 255;
      surface.data[(x + y * w) * 4 + 1] = b * 255;
      surface.data[(x + y * w) * 4 + 2] = b * 255;
      surface.vset(x, y, [ b, b, b ]);
    }
  }

  surface.update();
  return surface;
}

const imageMatToTrainingData = (a:Mat) => {
  const ti = Mat.alloc(a.rows * a.cols, 3);

  for (let x = 0; x < a.cols; x += 1) {
    for (let y = 0; y < a.rows; y += 1) {
      const row = x + y * a.rows;
      Mat.put(ti, row, 0, x/a.rows);
      Mat.put(ti, row, 1, y/a.cols);
      Mat.put(ti, row, 2, Mat.get(a, y, x));
    }
  }

  return ti;
}

const all = document.createElement('div');
all.style.position = 'absolute';
all.style.bottom = 0;
all.style.display = "flex";
all.style.alignItems = "flex-end";
document.body.appendChild(all);

const viewMatrix = (mat:Mat, s = 10, mask = [1,1,1]) => {

  const div = document.createElement('div');
  div.style.display = 'grid';
  div.style.gridTemplateColumns = `repeat(${mat.cols}, ${s}px)`;
  div.style.border = `2px solid ${rgb(mask.map(x => x * 255))}`;
  div.style.margin = `10px`;

  for (let x = 0; x < mat.cols; x += 1) {
    for (let y = 0; y < mat.rows; y += 1) {
      const b = Mat.at(mat, x, y) * 255;
      let p = document.createElement('div');
      p.style.backgroundColor = rgb([b *mask[0], b *mask[1], b *mask[2]]);
      p.style.width  = s + 'px';
      p.style.height = s + 'px';
      div.appendChild(p);
    }
  }

  all.appendChild(div);
}

const onTop = (el, scale = 1) => {
  el.style.position = 'fixed';
  el.style.bottom = 0;
  el.style.left = 0;
  el.style.zIndex = '1000';
  el.style.backgroundColor = 'black';
  el.style.transform = `scale(${scale})`;
  el.style.transformOrigin = 'bottom left';
  document.body.appendChild(el);
}

type Surface = {
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  data: Uint8ClampedArray,
  update: () => void,
  pset: (x:number, y:number, c:[number, number, number, number?]) => void,
  vset: (x:number, y:number, c:[number, number, number]) => void,
  pget: (x:number, y:number) => [number, number, number, number],
}

const newSurface = (w:number, h:number, img?):Surface => {
  const canvas = document.createElement('canvas');
  canvas.width  = w;
  canvas.height = h;

  const ctx = canvas.getContext('2d');
  if (img) ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, w, h);

  const pset = (x, y, c) => {
    imageData.data[(x + y * w) * 4 + 0] = c[0];
    imageData.data[(x + y * w) * 4 + 1] = c[1];
    imageData.data[(x + y * w) * 4 + 2] = c[2];
    imageData.data[(x + y * w) * 4 + 3] = c[3] ?? 255;
    ctx.putImageData(imageData, 0, 0);
  }

  const vset = (x, y, v) => {
    pset(x, y, v.map(n => n * 255));
  }

  return {
    canvas,
    pset,
    vset,
    data: imageData.data,
    update: () => ctx.putImageData(imageData, 0, 0),
    pget: (x, y) => {
      const i = (x + y * w) * 4;
      return [
        imageData.data[i + 0],
        imageData.data[i + 1],
        imageData.data[i + 2],
        imageData.data[i + 3]
      ];
    }
  }
}


//
// Main
//

export const main = async () => {

  console.clear();

  log.blue("Running MNIST Example");

  const imgA = await loadImage('/6.png');
  const a = imageToMatrix(imgA);

  const imgB = await loadImage('/9.png');
  const b = await imageToMatrix(imgB);


  // Training data (with stochastic shuffling)

  const trainData = imageMatToTrainingData(a);

  Mat.shuffleRows(trainData);

  const [ ti, to ] = Mat.splitCols(trainData, [ 2, 1 ]);


  // Train Network

  const net = NN.alloc([ 2, 7, 4, 7, 1 ], true);

  await train(net, ti, to, { 
    maxSteps: 20000,
    maxRank:  4,
    epochSize: 10,
    rate: 5
  }, imgA, 27, 27, 3);

}


main();


