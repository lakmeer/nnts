
//
// Deno / Vite Polymorhpic Imports
//
// In vite.config, we set a global define for VITE to true, which will therefore
// be undefined in Deno. This allows us to use the same code in both environments
// by using dymamic imports.
//

const Chalk = globalThis.VITE
  ? (await import("chalk")).default
  : await import("https://deno.land/std/fmt/colors.ts");

export { Chalk };


// This one is controlled by Vite aliases

export { AsciiTable3 } from 'npm:ascii-table3';

