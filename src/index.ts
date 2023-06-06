
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

//import './canvas'; // TODO: exports

import { logHelper, red, green, table, floor } from "./utils.ts";

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
    const ok = row.unshift() ? green("OK") : red("XX");
    rows.push([ ok, ...row ]);
  }

  if (!pass) log.red("⚠️  Failed to converge");

  const headers = [ 'OK' ].concat(inputCols).concat([ 'exp', 'act' ]);
  console.log(table(headers, rows));
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
  NN.train(xor, 0, 1, 10000, ti, to);

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


export const adder = (BITS = 2) => {

  log.blue("Running ADDER Example");

  // Generate training data

  const n = (1<<BITS);
  const rows = n*n;
  const ti = Mat.alloc(rows, BITS*2);
  const to = Mat.alloc(rows, BITS+1);

  for (let i = 0; i < ti.rows; i++) {
    let x = floor(i/n);
    let y = i%n;
    let z = x + y;

    let overflow = z >= n;

    for (let j = 0; j < BITS; j++) {
      Mat.put(ti, i, j,      (x >> j) & 1);
      Mat.put(ti, i, j+BITS, (y >> j) & 1);
      if (overflow) {
        Mat.put(to, i, j, 0);
      } else {
        Mat.put(to, i, j, (z >> j) & 1);
      }
    }
    Mat.put(to, i, BITS, overflow ? 1 : 0);
  }


  // Train Network

  const adder = NN.alloc([ BITS*2, 6, BITS+1 ], true);
  NN.train(adder, 0, 1, 10000, ti, to);

  // Report
  report(adder, ti, to, [ 'x', 'y' ], (inputs, expect, actual) => {
    const x = Mat.smush(Mat.sub(inputs, 0, 0, BITS, 1));
    const y = Mat.smush(Mat.sub(inputs, 0, BITS, BITS, 1));
    const exp = Mat.smush(expect, 0); // To nearest int
    const act = Mat.smush(actual, 0);
    const ok = exp == act;
    return [ ok, x, y, exp, act ];
  });
}


//
// Main
//

const main = () => {
  xor();
  adder();
}

main();

