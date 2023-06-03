

// Strings, Formatting

// Vite
//import C from "chalk";

// Deno
import * as C from "https://deno.land/std@0.187.0/fmt/colors.ts";

export const logHelper = (name: string, color: string) => {
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


// Math

export const rand = ():float => Math.random();


// Globalify

window.log = logHelper("NN");
window.rand = rand;

