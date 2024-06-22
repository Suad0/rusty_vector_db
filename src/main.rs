use std::collections::HashMap;

struct VectorStore {
    documents: Vec<String>,
    vectors: HashMap<String, Vec<f64>>,
}

impl VectorStore {
    fn new(documents: Vec<String>) -> Self {
        let mut store = VectorStore {
            documents: documents.clone(),
            vectors: HashMap::new(),
        };
        store.build();
        store
    }

    fn build(&mut self) {
        for doc in &self.documents {
            let vector = self.simple_encode(doc);
            self.vectors.insert(doc.clone(), vector);
        }
    }

    fn simple_encode(&self, doc: &str) -> Vec<f64> {
        // Simple encoding: counts of each character as a feature
        let mut vector = vec![0.0; 26];
        for ch in doc.chars().filter(|c| c.is_ascii_alphabetic()) {
            let idx = (ch.to_ascii_lowercase() as u8 - b'a') as usize;
            vector[idx] += 1.0;
        }
        vector
    }

    fn cosine_similarity(&self, u: &[f64], v: &[f64]) -> f64 {
        let dot_product: f64 = u.iter().zip(v).map(|(a, b)| a * b).sum();
        let norm_u: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_v: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        dot_product / (norm_u * norm_v)
    }

    fn get_top_n(&self, query: &str, n: usize) -> Vec<(String, f64)> {
        let embedded_query = self.simple_encode(query);
        let mut scores: Vec<(String, f64)> = self.vectors
            .iter()
            .map(|(doc, vec)| {
                let similarity = self.cosine_similarity(&embedded_query, vec);
                (doc.clone(), similarity)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.into_iter().take(n).collect()
    }
}

fn main() {
    let docs = vec![
        "I like apples".to_string(),
        "I like pears".to_string(),
        "I like dogs".to_string(),
        "I like cats".to_string(),
    ];

    let vs = VectorStore::new(docs);

    println!("{:?}", vs.get_top_n("I like apples", 1));
    println!("{:?}", vs.get_top_n("fruit", 2));
}
