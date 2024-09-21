use actix_files as fs;
use actix_web::{get, App, HttpResponse, HttpServer};
use clap::Parser;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

#[derive(Parser)]
#[clap(
    version = "1.0",
    author = "Your Name",
    about = "Compile Typst files into SVG_HTML presentations and serve them"
)]
struct Opts {
    #[clap(short, long, help = "Input Typst file")]
    entry: String,

    #[clap(long, help = "Watch for changes and recompile")]
    watch: bool,

    #[clap(long, default_value = "3000", help = "Port to serve the presentation")]
    port: u16,
}

#[get("/hello")]
async fn hello() -> HttpResponse {
    HttpResponse::Ok().body("Hello, World!")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let opts: Opts = Opts::parse();

    let input_path = Path::new(&opts.entry);
    if !input_path.exists() {
        eprintln!("Error: Input file '{}' does not exist.", opts.entry);
        std::process::exit(1);
    }

    compile_document(&opts.entry, opts.watch)?;

    let output_file = determine_output_path(input_path).expect("Failed to determine output path");
    let parent_dir = Arc::new(
        output_file
            .parent()
            .unwrap_or_else(|| Path::new("/"))
            .to_path_buf(),
    );
    let index_file = output_file
        .file_name()
        .and_then(|n| n.to_str())
        .expect("Failed to get index file name")
        .to_string();

    println!("Starting server at http://localhost:{}", opts.port);
    println!("Serving file: {}", output_file.display());

    HttpServer::new(move || {
        let parent_dir = Arc::clone(&parent_dir);
        App::new()
            .service(hello)
            .service(fs::Files::new("/", parent_dir.as_ref()).index_file(&index_file))
    })
    .bind(("127.0.0.1", opts.port))?
    .run()
    .await
}

fn compile_document(entry: &str, watch: bool) -> std::io::Result<()> {
    let mut cmd = Command::new("typst-ts-cli");
    cmd.arg("compile")
        .arg("-e")
        .arg(entry)
        .arg("--format")
        .arg("svg_html");

    if watch {
        cmd.arg("--watch");
    }

    println!("Compiling {}...", entry);
    let output = cmd.output()?;

    if !output.status.success() {
        eprintln!(
            "Compilation failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        std::process::exit(1);
    }

    println!(
        "Successfully compiled '{}' into a SVG_HTML presentation!",
        entry
    );
    Ok(())
}

fn determine_output_path(input_path: &Path) -> Option<PathBuf> {
    let file_stem = input_path.file_stem()?.to_str()?;
    let project_root = std::env::current_dir().ok()?;
    Some(project_root.join(format!("{}.artifact.svg.html", file_stem)))
}
