
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

import { weightColor, logHelper, sigmoid, red, green, table, floor, max, min, rgb, colorLerp } from "./utils.ts";

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


export const adder = (BITS = 2) => {

  log.blue("Running ADDER Example");

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


  // Train Network

  const adder = NN.alloc([ BITS*2, 4*BITS, BITS+1 ], true);
  NN.train(adder, 0, 1e1, 1000, ti, to);

  // Report
  report(adder, ti, to, [ 'x', 'y' ], (inputs, expect, actual) => {
    const x = Mat.smush(Mat.sub(inputs, 0, 0, BITS, 1), 0);
    const y = Mat.smush(Mat.sub(inputs, 0, BITS, BITS, 1), 0);
    const exp = Mat.smush(expect, 0);
    const act = Mat.smush(actual, 0);
    const ok = exp == act;
    return [ ok, x, y, exp, act ];
  });

  return {
    net: adder,
    cost: []
  }
}


//
// Main
//

const main = async () => {

  //const data = xor();
  const data = adder(2);


  //
  // Visualiser Testing
  //

  if (!globalThis.VITE) return;

  const Screen = await import('./canvas');

  Screen.setAspect(1);

  Screen.onFrame((ctx, { w, h }) => {

    // Background
    Screen.grid('grey', 0, 0, w, h, 20);
    Screen.grid('white', 0, 0, w, h, 2);

    // Padding
    const pad = 0; // w/20;
    Screen.zone(pad, pad, w - pad * 2, h - pad * 2, (ctx, { w, h }) => {

      // Net area
      Screen.zone(0, h/4, w/2, h/2, (ctx, { w, h }) => {

        const net = data.net;
        const numLayers = net.arch.length;
        const xStride = w/net.arch.length;

        const r = min(h/30, 0.9 * h / (2 * Math.max(...net.arch)));

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
      });
    });
  });

}

main();


