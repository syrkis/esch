// esch/src/app.ts
import { NodeCompiler } from "@myriaddreamin/typst-ts-node-compiler";
import fs from "node:fs/promises";
import http from "node:http";
import { JSDOM } from "jsdom";
import { watch } from "node:fs";
import { WebSocketServer, WebSocket } from "ws";
import path from "node:path";
import { fileURLToPath } from "node:url";
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
async function generateSlides(inputFile) {
    try {
        console.log("Starting slide generation...");
        const compiler = NodeCompiler.create();
        console.log("Compiler created.");
        console.log(`Reading ${inputFile}...`);
        const typstContent = await fs.readFile(inputFile, "utf-8");
        console.log(`${inputFile} read successfully.`);
        console.log("Compiling document...");
        const doc = compiler.compile({
            mainFileContent: typstContent,
        });
        console.log("Document compiled.");
        const outputDir = path.join(path.dirname(inputFile), "public", "slides");
        console.log(`Creating ${outputDir} directory...`);
        await fs.mkdir(outputDir, { recursive: true });
        console.log("Directory created.");
        console.log("Generating SVG...");
        const svg = compiler.svg({
            mainFileContent: typstContent,
        });
        const svgPath = path.join(outputDir, "slides.svg");
        console.log(`Writing ${svgPath}...`);
        await fs.writeFile(svgPath, svg);
        console.log("slides.svg written.");
        // Count the number of pages by parsing the SVG
        const dom = new JSDOM(svg);
        const pageCount = dom.window.document.querySelectorAll("svg > g").length;
        const metadataPath = path.join(outputDir, "metadata.json");
        console.log(`Writing ${metadataPath}...`);
        await fs.writeFile(metadataPath, JSON.stringify({
            generated: true,
            pageCount: pageCount,
            generatedAt: new Date().toISOString(),
        }));
        console.log("metadata.json written.");
        console.log("Evicting cache...");
        compiler.evictCache(10);
        console.log("Cache evicted.");
        console.log(`Slides generated successfully. Total pages: ${pageCount}`);
    }
    catch (error) {
        console.error("Error generating slides:", error);
        throw error;
    }
}
const PORT = 3000;
let wss;
async function startServer(inputFile) {
    try {
        await generateSlides(inputFile); // Generate slides initially
        const server = http.createServer(async (req, res) => {
            var _a, _b;
            try {
                if (req.url === "/") {
                    // Use the bundled index.html from the public directory
                    const indexPath = path.join(__dirname, "..", "public", "index.html");
                    const content = await fs.readFile(indexPath, "utf-8");
                    res.writeHead(200, { "Content-Type": "text/html" });
                    res.end(content);
                }
                else if ((_a = req.url) === null || _a === void 0 ? void 0 : _a.startsWith("/slides/metadata.json")) {
                    const metadataPath = path.join(path.dirname(inputFile), "public", "slides", "metadata.json");
                    const content = await fs.readFile(metadataPath, "utf-8");
                    res.writeHead(200, {
                        "Content-Type": "application/json",
                        "Cache-Control": "no-cache, no-store, must-revalidate",
                    });
                    res.end(content);
                }
                else if ((_b = req.url) === null || _b === void 0 ? void 0 : _b.startsWith("/slides/slides.svg")) {
                    const svgPath = path.join(path.dirname(inputFile), "public", "slides", "slides.svg");
                    const content = await fs.readFile(svgPath, "utf-8");
                    res.writeHead(200, {
                        "Content-Type": "image/svg+xml",
                        "Cache-Control": "no-cache, no-store, must-revalidate",
                    });
                    res.end(content);
                }
                else {
                    res.writeHead(404, { "Content-Type": "text/plain" });
                    res.end("Not Found");
                }
            }
            catch (error) {
                console.error("Error serving request:", error);
                res.writeHead(500, { "Content-Type": "text/plain" });
                res.end("Internal Server Error");
            }
        });
        server.listen(PORT, () => {
            console.log(`Server running at http://localhost:${PORT}/`);
        });
        // Setup WebSocket server
        wss = new WebSocketServer({ server });
        // Watch for changes to the input file
        watch(inputFile, async (eventType, filename) => {
            if (eventType === "change") {
                console.log(`${inputFile} has changed. Regenerating slides...`);
                try {
                    await generateSlides(inputFile);
                    console.log("Slides regenerated successfully.");
                    // Notify all connected clients to reload
                    for (const client of wss.clients) {
                        if (client.readyState === WebSocket.OPEN) {
                            client.send("reload");
                        }
                    }
                }
                catch (error) {
                    console.error("Error regenerating slides:", error);
                }
            }
        });
    }
    catch (error) {
        console.error("Failed to start server:", error);
    }
}
export async function main(inputFile) {
    await startServer(inputFile);
}
// If running directly (not imported), start the server
if (import.meta.url === `file://${process.argv[1]}`) {
    const inputFile = process.argv[2];
    if (!inputFile) {
        console.error("Please provide a Typst file as an argument.");
        process.exit(1);
    }
    main(inputFile);
}
