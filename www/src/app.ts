import { NodeCompiler } from "@myriaddreamin/typst-ts-node-compiler";
import fs from "node:fs/promises";
import path from "node:path";
import http from "node:http";
import { JSDOM } from "jsdom";

async function generateSlides() {
  try {
    console.log("Starting slide generation...");
    const compiler = NodeCompiler.create();
    console.log("Compiler created.");

    console.log("Reading main.typ...");
    const typstContent = await fs.readFile("main.typ", "utf-8");
    console.log("main.typ read successfully.");

    console.log("Compiling document...");
    const doc = compiler.compile({
      mainFileContent: typstContent,
    });
    console.log("Document compiled.");

    console.log("Creating public/slides directory...");
    await fs.mkdir("public/slides", { recursive: true });
    console.log("Directory created.");

    console.log("Generating SVG...");
    const svg = compiler.svg({
      mainFileContent: typstContent,
    });

    console.log("Writing slides.svg...");
    await fs.writeFile("public/slides/slides.svg", svg);
    console.log("slides.svg written.");

    // Count the number of pages by parsing the SVG
    const dom = new JSDOM(svg);
    const pageCount = dom.window.document.querySelectorAll("svg > g").length;

    console.log("Writing metadata.json...");
    await fs.writeFile(
      "public/slides/metadata.json",
      JSON.stringify({
        generated: true,
        pageCount: pageCount,
        generatedAt: new Date().toISOString(),
      }),
    );
    console.log("metadata.json written.");

    console.log("Evicting cache...");
    compiler.evictCache(10);
    console.log("Cache evicted.");

    console.log(`Slides generated successfully. Total pages: ${pageCount}`);
  } catch (error) {
    console.error("Error generating slides:", error);
    throw error;
  }
}

const PORT = 3000;

async function startServer() {
  try {
    await generateSlides(); // Generate slides before starting the server

    const server = http.createServer(async (req, res) => {
      try {
        if (req.url === "/") {
          const content = await fs.readFile("public/index.html", "utf-8");
          res.writeHead(200, { "Content-Type": "text/html" });
          res.end(content);
        } else if (req.url === "/slides/metadata.json") {
          const content = await fs.readFile("public/slides/metadata.json", "utf-8");
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(content);
        } else if (req.url === "/slides/slides.svg") {
          const content = await fs.readFile("public/slides/slides.svg", "utf-8");
          res.writeHead(200, { "Content-Type": "image/svg+xml" });
          res.end(content);
        } else {
          res.writeHead(404, { "Content-Type": "text/plain" });
          res.end("Not Found");
        }
      } catch (error) {
        console.error("Error serving request:", error);
        res.writeHead(500, { "Content-Type": "text/plain" });
        res.end("Internal Server Error");
      }
    });

    server.listen(PORT, () => {
      console.log(`Server running at http://localhost:${PORT}/`);
    });
  } catch (error) {
    console.error("Failed to start server:", error);
  }
}

startServer();
