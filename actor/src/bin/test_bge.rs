use actor::*;
use std::time::Instant;

fn main() {
    let onnx = onnx::Onnx::new(18);
    println!("Loading BGE model...");
    let mut bge = bge::Bge::new(&onnx, onnx::Executor::Cpu).expect("Failed to load BGE model");
    println!("Model loaded.");

    // Embed concept labels
    let concepts = [
        "greeting",
        "statement",
        "question",
        "happy",
        "frustrated",
        "sad",
        "angry",
        "sarcastic",
    ];

    let start = Instant::now();
    let concept_embeddings: Vec<bge::Embedding> = concepts
        .iter()
        .map(|c| bge.embed(c).expect("Failed to embed concept"))
        .collect();
    let concept_time = start.elapsed();
    println!(
        "Embedded {} concepts in {:.2}ms",
        concepts.len(),
        concept_time.as_secs_f64() * 1000.0
    );

    // Embed test sentence
    let test_sentence = "Hello there! Good morning!";
    let start = Instant::now();
    let test_embedding = bge.embed(test_sentence).expect("Failed to embed test sentence");
    let embed_time = start.elapsed();

    // Compare against concepts
    let start = Instant::now();
    let scores = test_embedding.similarities(&concept_embeddings);
    let compare_time = start.elapsed();

    println!("\nTest sentence: \"{}\"\n", test_sentence);
    println!("{:<12} {:>8}", "Concept", "Score");
    println!("{}", "-".repeat(22));
    for (concept, score) in concepts.iter().zip(scores.iter()) {
        println!("{:<12} {:>8.4}", concept, score);
    }

    println!(
        "\nEmbed time:   {:.2}ms",
        embed_time.as_secs_f64() * 1000.0
    );
    println!(
        "Compare time: {:.4}ms",
        compare_time.as_secs_f64() * 1000.0
    );
}
