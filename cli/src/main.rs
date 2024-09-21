use actix_files as fs;
use actix_web::{App, HttpServer};
use clap::Parser;
use ctrlc;
use std::fs::remove_file;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::{Arc, Mutex};

#[derive(Parser)]
struct Opts {
    #[clap(short, long, help = "Input Typst file")]
    entry: String,

    #[clap(long, default_value = "3000", help = "Port to serve the presentation")]
    port: u16,
}

#[actix_web::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts: Opts = Opts::parse();

    let input_path = Path::new(&opts.entry);
    if !input_path.exists() {
        return Err(format!("Error: Input file '{}' does not exist.", opts.entry).into());
    }

    let child_process = Arc::new(Mutex::new(compile_document(&opts.entry)?));

    let output_file = determine_output_path(input_path)?;
    let parent_dir = Arc::new(
        output_file
            .parent()
            .unwrap_or_else(|| Path::new("/"))
            .to_path_buf(),
    );
    let index_file = output_file
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or("Failed to get index file name")?
        .to_string();

    println!("Starting server at http://localhost:{}", opts.port);
    println!("Serving file: {}", output_file.display());

    // Set up Ctrl+C handler
    let output_file_clone = output_file.clone();
    let child_process_clone = Arc::clone(&child_process);
    ctrlc::set_handler(move || {
        println!("Shutting down...");
        if let Ok(mut child) = child_process_clone.lock() {
            let _ = child.kill();
        }
        if let Err(e) = remove_file(&output_file_clone) {
            eprintln!("Failed to remove output file: {}", e);
        }
        std::process::exit(0);
    })?;

    HttpServer::new(move || {
        let parent_dir = Arc::clone(&parent_dir);
        App::new().service(
            fs::Files::new("/", parent_dir.as_ref())
                .index_file(&index_file)
                .use_etag(false)
                .use_last_modified(false),
        )
    })
    .bind(("127.0.0.1", opts.port))?
    .run()
    .await?;

    Ok(())
}

fn compile_document(entry: &str) -> Result<Child, std::io::Error> {
    let mut cmd = Command::new("typst-ts-cli");
    cmd.arg("compile")
        .arg("--entry")
        .arg(entry)
        .arg("--format")
        .arg("svg_html")
        .arg("--watch");

    println!("Compiling {} and watching for changes...", entry);

    let child = cmd.spawn()?;

    println!(
        "Successfully started compilation of '{}' into a SVG_HTML presentation!",
        entry
    );
    Ok(child)
}

fn determine_output_path(input_path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let file_stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("Invalid input file name")?;
    let project_root = std::env::current_dir()?;
    Ok(project_root.join(format!("{}.artifact.svg.html", file_stem)))
}
