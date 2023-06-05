
// Shim C types

declare global {
  type float = number;
  type int   = number;
}


// Strings, Formatting

//import C from "chalk"; // Vite
import * as C from "https://deno.land/std@0.187.0/fmt/colors.ts"; // Deno

type LogFn = (...args: any[]) => void;
type LogHlp = { [key: string]: LogFn }

export const logHelper = (name: string):LogHlp => {
  const label = `[${name}]`;
  return {
    ok:    (...args) => console.log(C.green(label), ...args),
    info:  (...args) => console.log(C.blue(label), ...args),
    err:   (...args) => console.log(C.red(label), ...args),
    quiet: (...args) => console.log(C.dim([label, ...args].join(' '))),
    red:   (...args) => console.log(C.red([label, ...args].join(' '))),
    green: (...args) => console.log(C.green([label, ...args].join(' '))),
    blue:  (...args) => console.log(C.blue([label, ...args].join(' '))),
  }
}

export const log = logHelper("NN");
export const { red, green } = C;


// Math

export const { min, max, sin, cos, exp, log10, pow, sqrt, abs, floor, PI } = Math;
export const ln = Math.log; // Already used this name

export const rand = ():float => Math.random();
export const limit = (a:number, b:number, n:number) => min(b, max(a, n));


// Text-mode Tables

// import { AsciiTable3 } from 'ascii-table3' // Vite
import { AsciiTable3 } from 'npm:ascii-table3'; // Deno
export { AsciiTable3 };

export const table = (headers:Array<string>, rows:Array<Array<any>>):string => {
  const t = new AsciiTable3().setStyle('unicode-round');
  for (let i = 0; i < headers.length; i++) t.setAlignCenter(i + 1); 
  t.setHeading(...headers);
  t.addRowMatrix(rows);
  return t.toString();
}


// Misc

export const assert = (cond:boolean, msg:string) => {
  if (!cond) throw new Error(msg);
}


