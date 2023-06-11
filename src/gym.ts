
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

type TrainState =
  | "IDLE"
  | "TRAINING"
  | "STALLED"
  | "STOPPED"
  | "FINISHED";

export const train = async (net:NN, ti, to, options = {}) => {

  const maxSteps   = options.maxSteps  ?? 10000;
  const maxRank    = options.maxRank   ?? 4;
  const rate       = options.rate      ?? 1;
  const epochSize  = options.epochSize ?? 100;
  const aggression = options.aggression ?? 0;
  const jitter     = options.jitter    ?? 0;

  const maxWindow = 100;

  Screen.setAspect(1);


  // Prepare Gradient Network

  const grad = NN.alloc(net.arch);

  log.info(`Training for ${maxSteps} steps...`);

  let c = 1;
  let rank = 0;
  let costHist = [];
  let step = 0;
  let state: TrainState = "TRAINING";


  // Training loop

  const trainBatch = async () => {

    //const r = rate;
    const a = limit(1, 10, 1/(c + 1 - aggression * (step/maxSteps)));
    const r = rate/2 + rate/2 * a; // scale learning rate as cost drops

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

    // Draw new frame
    Screen.all((ctx, { w, h }) => {
      Screen.clear();

      Screen.grid('grey', 0, 0, w, h, 16);

      Screen.zone(0, h/4, w, h*3/4, (ctx, size) => {
        Screen.pad(size, 0.9, (ctx, size) => {
          Screen.drawNetwork(grad, size, 1.2, 10, true);
          Screen.drawNetwork(net,  size, 1,   1);
        });
      });

      Screen.zone(0, 0, w, h/4, (ctx, size) => {
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

}



