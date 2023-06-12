
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

const AUTOBLEND = false;

export const train = async (net:NN, trainingSet, options = {}, imgA, imgB, w, h) => {

  const maxEpochs  = options.maxEpochs  ?? 10000;
  const maxRank    = options.maxRank    ?? 4;
  const rate       = options.rate       ?? 1;
  const batchesPF  = options.batchesPF  ?? 100;
  const batchSize  = options.batchSize  ?? 100;
  const aggression = options.aggression ?? 0;
  const jitter     = options.jitter     ?? 0;
  const upSize     = options.upSize     ?? 1;


  // Prepare Gradient Network

  const grad = NN.alloc(net.arch);

  log.info(`Training for ${maxEpochs} epochs...`);


  // State

  let c = 1;
  let costHist = [];
  let rank = 0;
  let epoch = 0;
  let frame = 0;

  let state: TrainState = "TRAINING";

  const inputA  = newSurface(w, h, imgA);
  const inputB  = newSurface(w, h, imgB);
  const outputA = newSurface(w, h);
  const outputB = newSurface(w, h);

  const upscale = newSurface(w * upSize, h * upSize);


  // Batching

  let batchStart = 0;
  let blend      = 0.5;
  let avgCost    = 0;

  log.info(`${trainingSet.rows} training samples in ${Math.ceil(trainingSet.rows/batchSize)} batches of ${batchSize}`);


  // Interaction

  const cancel = () => state = "CANCELLED";
  document.addEventListener('keydown', cancel);

  if (!AUTOBLEND) {
    document.addEventListener('mousemove', ({ clientX }) => {
      blend = clientX / window.innerWidth;
    });
  }

  // Training loop (one frame)

  const trainFrame = async () => {

    // Scale learning rate as we progress
    const a = limit(1, 10, 1/(c + 1 - aggression * (epoch/maxEpochs)));
    const r = rate/2 + rate/2 * a;

    // Apply backpropagated gradient in batches
    for (let i = 0; i < batchesPF; i++) {

      let size = batchSize;

      if (batchStart + batchSize >= trainingSet.rows) {
        size = trainingSet.rows - batchStart;
      }

      const batch_ti = Mat.sub(trainingSet, batchStart, 0, size, 3);
      const batch_to = Mat.sub(trainingSet, batchStart, 3, size, 1);

      NN.backprop(net, grad, 0, batch_ti, batch_to);
      NN.learn(net, grad, r);
      avgCost += NN.cost(net, batch_ti, batch_to);

      batchStart += size;

      if (batchStart >= trainingSet.rows) {

        // Reset for new epoch
        epoch += 1;
        batchStart = 0;

        // Commit avg cost to series data
        c = avgCost / batchesPF;
        costHist.push(c);
        avgCost = 0;

        // New shuffle orger for next epoch
        Mat.shuffleRows(trainingSet);
      }
    }

    // Review frame
    rank = costRank(c);

    if (epoch >= maxEpochs) {
      log.err("Stopping: Max epochs reached");
      state = "STOPPED";
    }

    if (rank  >= maxRank) {
      log.ok("Finished: Cost rank goal reached -", rank, costRank(c, true));
      state = "FINISHED";
    }


    // Autoblend

    if (AUTOBLEND) {
      blend = Math.cos(performance.now() / 2000) / 2 + 0.5;
    }


    // Run inference and paint to canvas(es)

    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        const i = (x + y * h);

        // Load pixels
        Mat.put(net.as[0], 0, 0, x/w);
        Mat.put(net.as[0], 0, 1, y/h);

        // Run for blend = 0
        Mat.put(net.as[0], 0, 2, 0);
        NN.forward(net);
        const a = abs(Mat.at(net.as[net.count], 0, 0));
        outputA.vset(x, y, [ a, a, a ]);

        // Run for blend = 1
        Mat.put(net.as[0], 0, 2, 1);
        NN.forward(net);
        const b = abs(Mat.at(net.as[net.count], 0, 0));
        outputB.vset(x, y, [ b, b, b ]);

        //diff.pset(x, y, weightColor(expect - b, 1, true));
      }
    }

    // Commit pixels to surface
    outputA.update();
    outputB.update();
    //diff.update();

    // Only render upscale of factor n every nth frame
    if ((frame % upSize) === 0) {
      Mat.put(net.as[0], 0, 2, blend);
      netToImg(net, upscale, w, h, upSize);
    }

    // Draw new frame
    Screen.all((ctx, { w, h }) => {
      Screen.clear();

      Screen.grid('grey', 0, 0, w, h, 8);

      // Network diagram
      Screen.zone(w/4, h/4, w*1/2, h*1/2, (ctx, size) => {
        Screen.pad(size, 0.9, (ctx, size) => {
          Screen.drawNetwork(grad, size, 1.2, 10, true);
          Screen.drawNetwork(net,  size, 1,   1);
        });
      });

      // Input image A
      Screen.zone(0, h*1/4, w/4, h/4, (ctx, size) => {
        Screen.pad(size, 0.9, (ctx, size) => {
          Screen.image(inputA.canvas, size);
        });
      });

      // Output image A
      Screen.zone(0, h*2/4, w/4, h/4, (ctx, size) => {
        Screen.pad(size, 0.9, (ctx, size) => {
          Screen.image(outputA.canvas, size);
        });
      });

      // Input image B
      Screen.zone(w*3/4, h*1/4, w/4, h/4, (ctx, size) => {
        Screen.pad(size, 0.9, (ctx, size) => {
          Screen.image(inputB.canvas, size);
        });
      });

      // Output image B
      Screen.zone(w*3/4, h*1/2, w/4, h/4, (ctx, size) => {
        Screen.pad(size, 0.9, (ctx, size) => {
          Screen.image(outputB.canvas, size);
        });
      });

      // Blended image
      Screen.zone(3/4 * w*blend, h*3/4, w/4, h/4, (ctx, size) => {
        Screen.pad(size, 0.9, (ctx, size) => {
          Screen.image(upscale.canvas, size);
          Screen.text(`${(blend * 100).toFixed(0)}%`, 10, 25, 'white', 22);
        });
      });


      // Cost history plot
      Screen.zone(0, 0, w, h/4, (ctx, size) => {
        Screen.plotSeries(costHist, size, options.color ?? '#ff2266', true);

        const tCost = c.toFixed(maxRank - 1);
        const tRate = r.toFixed(2);
        const tTime = ((performance.now() - start)/1000).toFixed(2);
        const tRank = costRank(c, true);

        Screen.text(`Cost: ${tCost}  Rate: ${tRate}  Epoch: ${epoch}/${maxEpochs}  ${tTime}s`, 10, 25, 'white', 22);
        Screen.text(`${state} ${tRank}`, w - 160, 25, 'white', 22);
        Screen.text(`Set size: ${trainingSet.rows}`, 10, size.h - 10, 'white', 22);
      });
    });

    // Stop if we're not improving
    if (state === "TRAINING") {
      await new Promise(requestAnimationFrame);
      await trainFrame();
    }

    frame += 1;

    // Gym might be interested in the final gradient network
    return grad;
  }


  // Begin training

  const start = performance.now();
  await trainFrame();
  const time = performance.now() - start;

  if (rank < maxRank) {
    log.err(`Stopping at rank ${rank} after ${epoch} epochs.`);
  } else {
    log.ok(`Finished in ${time.toFixed(2)}ms and ${epoch} epochs`);
  }

  document.removeEventListener('keydown', cancel);
}



//
// Image Training Helpers
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
  const mat = Mat.alloc(a.rows * a.cols, 3);

  for (let x = 0; x < a.cols; x += 1) {
    for (let y = 0; y < a.rows; y += 1) {
      const row = x + y * a.rows;
      Mat.put(mat, row, 0, x/a.rows);
      Mat.put(mat, row, 1, y/a.cols);
      Mat.put(mat, row, 2, Mat.get(a, y, x));
    }
  }

  return mat;
}

const blendedImagesTrainingData = (a:Mat, b:Mat) => {

  if (a.rows !== b.rows || a.cols !== b.cols) throw new Error("Blending images must have same dimensions");

  // [ #pixels ] x [ x, y, blend, output ]
  const mat = Mat.alloc(a.rows * a.cols * 2, 4);

  for (let z = 0; z < 2; z += 1) {
    for (let x = 0; x < a.cols; x += 1) {
      for (let y = 0; y < a.rows; y += 1) {
        const row = x + y * a.rows + z * a.rows * a.cols;
        Mat.put(mat, row, 0, x/a.cols);
        Mat.put(mat, row, 1, y/a.rows);
        Mat.put(mat, row, 2, z);
        Mat.put(mat, row, 3, Mat.at(z ? b : a, y, x));
      }
    }
  }

  return mat;
}


//
// Debug features
//

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

  Screen.setAspect(1);

  log.blue("Running MNIST Blending Example");

  const imgA = await loadImage('/6.png');
  const a = imageToMatrix(imgA);

  const imgB = await loadImage('/9.png');
  const b = await imageToMatrix(imgB);


  // Training data

  const trainData = blendedImagesTrainingData(a, b);


  // Train Network

  const net = NN.alloc([ 3, 7, 6, 5, 1 ], true);

  await train(net, trainData, {
    maxEpochs: 20000,
    maxRank: 5,
    batchesPF: 60,
    batchSize: 100,
    rate: 1,
    upSize: 6,
  }, imgA, imgB, 27, 27);

}


main();


