
//
// NN.ts
//
// Reimplemented from Tsoding (https://twitch.tv/tsoding)
// by watching recordings (https://youtube.com/@TsodingDaily),
// not by copying the code (https://github.com/tsoding/nn.h).
//
// This is just to scratch my own itch, do not use lol.
//


// Custom imports

import { pad, costRank, weightColor, logHelper, sigmoid, red, green, table, floor, max, min, rgb, colorLerp } from "./utils.ts";
import * as Screen from './canvas';

const log = logHelper("GYM");


// NN Implementation

import * as NN from  "./nn.ts";
import * as Mat from "./matrix.ts";
import type { Net } from "./nn.ts";
import type { Matrix } from "./matrix.ts";


//
// Training report formatter
//

type ReportRow = [ boolean, ...string[] ];
type ReportRowFormatter = (inputs:Matrix, expect:Matrix, actual:Matrix) => ReportRow;

const report = (net:Net, ti: Matrix, to:Matrix, inputCols:string[], formatRow:ReportRowFormatter):boolean => {
  const input  = net.as[0];
  const output = net.as[net.count];
  const c = NN.cost(net, ti, to);

  log.info(`Final Cost:`, c);

  const rows = [];
  let pass = true;

  for (let ix = 0; ix < to.rows; ix++) {
    Mat.copy(input, Mat.row(ti, ix));
    NN.forward(net);

    const row = formatRow(input, Mat.row(to, ix), Mat.row(output, 0));
    pass = pass && row[0];
    const ok = row.shift() ? green("OK") : red("XX");
    rows.push([ ok, ...row ]);
  }

  const headers = [ 'OK' ].concat(inputCols).concat([ 'exp', 'act' ]);

  if (!pass) {
    console.log(table(headers, rows));
    log.red("⚠️  Failed to converge");
  }
  return pass;
}


//
// Examples
//

export const xor = () => {

  log.blue("Running XOR Example");

  // XOR Training data
  const ti = Mat.create(4, 2, [ 0, 0,    0, 1,    1, 0,    1, 1   ]);
  const to = Mat.create(4, 1, [       1,       0,       0,      1 ]);

  // Train XOR Network
  const xor = NN.alloc([ 2, 2, 1 ], true);
  const cost = NN.train(xor, 0, 1, 10000, ti, to);

  // Report
  report(xor, ti, to, [ 'a', 'b' ], (inputs, expect, actual) => {
    return [
      Mat.smush(expect, 0) == Mat.smush(actual, 0),
      Mat.at(inputs, 0, 0).toString(),
      Mat.at(inputs, 0, 1).toString(),
      Mat.smush(expect, 0),
      Mat.smush(actual, 0)
    ]
  });

  return {
    net: xor,
    cost: cost,
  }

}


export const adder = async (BITS = 2) => {

  log.blue("Running ADDER Example");

  Screen.setAspect(1);


  // Generate training data

  const n = (1<<BITS);
  const rows = n*n;
  const ti = Mat.alloc(rows, BITS*2);
  const to = Mat.alloc(rows, BITS+1);

  for (let i = 0; i < ti.rows; i++) {
    let x = i/n | 0;
    let y = i%n | 0;
    let z = x + y;

    for (let j = 0; j < BITS; j++) {
      Mat.put(ti, i, j,      (x >> j) & 1);
      Mat.put(ti, i, j+BITS, (y >> j) & 1);
      Mat.put(to, i, j,      (z >> j) & 1);
    }
    Mat.put(to, i, BITS, z >= n ? 1 : 0);
  }


  // Prepare Network

  const net = NN.alloc([ BITS*2, 4*BITS, 3*BITS, BITS+1 ], true);
  const grad = NN.alloc(net.arch);

  const maxSteps = 10000;
  const maxRank  = 4;  // Number of zeroes before it's good enough
  const rate = 1;
  const batchSize = 100;

  log.info(`Training for ${maxSteps} steps...`);

  let c = 0;
  let step = 0;


  // Training loop

  const trainBatch = async () => {
    for (let i = 0; i < batchSize; i++) {
      NN.backprop(net, grad, 0, ti, to);
      NN.learn(net, grad, rate);
    }

    step += batchSize;

    // Draw new frame
    Screen.all((ctx, { w, h }) => {
      Screen.clear();
      Screen.grid('grey',  0, 0, w, h, 20);
      Screen.grid('white', 0, 0, w, h, 2);
      Screen.zone(0, h/16, w, h - h/8, (ctx, size) => drawNetwork(net, ctx, size));
      //Screen.zone(0, h/4, w/2, h/2, (ctx, size) => drawNetwork(net, ctx, size));
    });

    c = NN.cost(net, ti, to);
    log.quiet(`[${pad(6, '#' + step)}] ${costRank(c, true)} ${c}`);

    if (step < maxSteps && costRank(c) <= maxRank) {
      await new Promise(requestAnimationFrame);
      await trainBatch();
    }
  }


  // Begin training

  const start = performance.now();
  await trainBatch();
  const time = performance.now() - start;
  const rank = costRank(c);
  log.info(`${rank} Finished in ${time.toFixed(2)}ms`);


  // Report

  report(net, ti, to, [ 'x', 'y' ], (inputs, expect, actual) => {
    const x = Mat.smush(Mat.sub(inputs, 0, 0, BITS, 1), 0);
    const y = Mat.smush(Mat.sub(inputs, 0, BITS, BITS, 1), 0);
    const exp = Mat.smush(expect, 0);
    const act = Mat.smush(actual, 0);
    const ok = exp == act;
    return [ ok, x, y, exp, act ];
  });

  return adder;
}


//
// Draw a Network
//

const drawNetwork = (net:Net, ctx, { w, h }) => {
  const numLayers = net.arch.length;
  const xStride = w/net.arch.length;

  const r = min(h/35, 0.9 * h / (2 * Math.max(...net.arch)));

  // Each layer
  for (let layer = 0; layer < numLayers; layer++) {
    const numNeurons = net.arch[layer];
    const yStride = h/numNeurons;
    const x = xStride/2 + layer * xStride;
    const c = `hsl(${ 360/numLayers * layer }, 100%, 50%)`;

    // Each neuron in layer
    for (let i = 0; i < numNeurons; i++) {
      const y = yStride/2 + i * yStride;

      // Draw connections to next layer
      if (layer < numLayers - 1) {
        const nextNumNeurons = net.arch[layer+1];
        const nextYStride = h/nextNumNeurons;

        for (let j = 0; j < nextNumNeurons; j++) {
          const w = Mat.at(net.ws[layer], i, j);
          const nextY = nextYStride/2 + j * nextYStride;
          Screen.line(rgb(weightColor(w)), x, y, x+xStride, nextY);
        }
      }

      // Draw neuron
      const b = layer == 0 ? 0 : Mat.at(net.bs[layer-1], 0, i);
      Screen.circle(rgb(weightColor(b)), x, y, r);
    }
  }
}


//
// Main
//

const main = () => {
  adder(3);
}

main();


