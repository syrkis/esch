#!/usr/bin/env node

import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { main } from "../dist/app.js";

const inputFile = process.argv[2];
if (!inputFile) {
  console.error("Please provide a Typst file as an argument.");
  process.exit(1);
}

const resolvedPath = resolve(process.cwd(), inputFile);
main(resolvedPath);
