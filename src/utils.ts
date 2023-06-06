
import { Chalk as C, AsciiTable3 } from "./_poly-import.ts";


// Shim C types

declare global {
  type float = number;
  type int   = number;
}


// Strings, Formatting

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
export const pad = (len:number, str:string) => str.padStart(len, ' ');


// Arrays

export const last = <T>(arr:Array<T>):T => arr[arr.length - 1];


// Math

export const { min, max, sin, cos, exp, log10, pow, sqrt, abs, floor, PI } = Math;
export const ln = Math.log; // Already used this name

export const rand = ():float => Math.random();
export const limit = (a:number, b:number, n:number) => min(b, max(a, n));


// Text-mode Tables

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

export const costRank = (n:float) => {
  const rank = -floor(log10(n));
  return [ "🔴", "🟠", "🟡", "🟢", "🔵" ][limit(0, 4, rank - 1)];
}


