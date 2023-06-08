
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

import { limit, unbend, abs, pad, last, costRank, weightColor, logHelper, sigmoid, red, green, table, floor, max, min, rgb, colorLerp } from "./utils.ts";
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
//
// Training Loop
//

export const train = async (net:NN, ti, to, options = {}) => {

  const maxSteps  = options.maxSteps  ?? 10000;
  const maxRank   = options.maxRank   ?? 4;
  const rate      = options.rate      ?? 1;
  const batchSize = options.batchSize ?? 100;
  const aggression = options.aggression ?? 0;

  Screen.setAspect(1);


  // Prepare Gradient Network

  const grad = NN.alloc(net.arch);

  log.info(`Training for ${maxSteps} steps...`);

  let c = 1;
  let costHist = [];
  let step = 0;

  //let g = 0;
  //let gradHist = [];


  // Training loop

  const trainBatch = async () => {

    //const r = rate;
    const a = limit(1, 10, 1/(c + 1 - aggression * (step/maxSteps)));
    const r = rate/2 + rate/2 * a; // scale learning rate as cost drops

    // Apply backpropagated gradient
    for (let i = 0; i < batchSize; i++) {
      NN.backprop(net, grad, 0, ti, to);
      NN.learn(net, grad, r);
    }

    // Review batch
    step += batchSize;
    c = NN.cost(net, ti, to);
    costHist.push(c);

    // "Gradient velocity"
    //NN.forward(grad);
    //g =  grad.ws.reduce((sum, mat) => mat.data.reduce((sum, v) => sum + abs(v), sum), 0);
    //g += grad.bs.reduce((sum, mat) => mat.data.reduce((sum, v) => sum + abs(v), sum), 0);
    //g += grad.as.reduce((sum, mat) => mat.data.reduce((sum, v) => sum + abs(v), sum), 0);
    //gradHist.push(abs(nc - c));

    const running = step < maxSteps && costRank(c) <= maxRank;

    // Draw new frame
    Screen.all((ctx, { w, h }) => {

      Screen.clear();

      Screen.grid('grey',  0, 0, w, h, 16);
      //Screen.grid('white', 0, 0, w, h, 2);

      Screen.zone(0, h/4, w, h*3/4, (ctx, size) => {
        Screen.pad(size, 0.9, (ctx, size) => {
          Screen.drawNetwork(grad, size, 1.2, 10000, true);
          Screen.drawNetwork(net,  size, 1,   10);
        });
      });

      Screen.zone(0, 0, w, h/4, (ctx, size) => {
        //Screen.plotSeriesLog(gradHist, size, 'limegreen');
        Screen.plotSeriesLog(costHist, size, '#ff3355');
      });

      Screen.text(`Cost: ${c.toFixed(7)}  Rate: ${r.toFixed(4)}  Step: ${step}/${maxSteps}`, 10, 25, 'white', 22);
      Screen.text(`${running ? 'TRAINING' : 'FINISHED' } ${costRank(c, true)}`, w - 160, 25, 'white', 22);
    });

    // Stop if we're not improving
    if (running) {
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
  const rank = costRank(c);

  if (rank < maxRank) {
    log.err(`Stopping at rank ${rank} after ${step} steps.`);
  } else {
    log.ok(`Finished in ${time.toFixed(2)}ms and ${step} steps`);
  }

}


//
// Examples
//

export const xor = async () => {

  log.blue("Running XOR Example");


  // XOR Training data

  const ti = Mat.create(4, 2, [ 0, 0,    0, 1,    1, 0,    1, 1   ]);
  const to = Mat.create(4, 1, [       1,       0,       0,      1 ]);


  // Train XOR Network

  const xor = NN.alloc([ 2, 2, 1 ], true);

  await train(xor, ti, to, { rate: 0.5 });


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
}


const adder = async (BITS:number) => {

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


  // Train network


  const net = NN.alloc([ BITS*2, 4*BITS, 3*BITS, BITS+1 ], true);

  await train(net, ti, to, {
    maxSteps:  10000,
    maxRank:   3,
    rate:      5,
    batchSize: 10,
    offset:    0.0,
    color: 'limegreen',
    aggression: 0.0
  });


  // Report

  report(net, ti, to, [ 'x', 'y' ], (inputs, expect, actual) => {
    const x = Mat.smush(Mat.sub(inputs, 0, 0, BITS, 1), 0);
    const y = Mat.smush(Mat.sub(inputs, 0, BITS, BITS, 1), 0);
    const exp = Mat.smush(expect, 0);
    const act = Mat.smush(actual, 0);
    const ok = exp == act;
    return [ ok, x, y, exp, act ];
  });

  return net;
}



//
// Main
//

export const main = () => {
  //xor();
  adder(4);
}

main();

