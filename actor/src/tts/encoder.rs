use {
    super::*,
    rand_distr::{Distribution, Normal},
    std::{collections::HashMap, sync::Arc},
};

pub struct Encoder {
    flow_main_session: onnx::Session,
    flow_step_session: onnx::Session,
    conditioner_session: onnx::Session,
    input_names: Vec<String>,
    output_names: Vec<String>,
    reset_state_tensors: Vec<onnx::Value>,
    state_tensors: Vec<onnx::Value>,
    sequence_tensor: onnx::Value,
    s_tensor: onnx::Value,
    t_tensor: onnx::Value,
    c_tensor: onnx::Value,
    x_tensor: onnx::Value,
}

impl Encoder {
    pub fn new(onnx: &Arc<onnx::Onnx>, executor: onnx::Executor) -> Self {
        let flow_main_session = onnx.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, FLOW_MAIN_PATH);
        let flow_step_session = onnx.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, FLOW_STEP_PATH);
        let conditioner_session = onnx.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, CONDITIONER_PATH);

        // extract input and output names
        let mut input_names = Vec::<String>::new();
        let input_count = flow_main_session.input_count();
        for i in 0..input_count {
            let name = flow_main_session.input_name(i);
            if !["sequence", "text_embeddings"].contains(&name.as_str()) {
                input_names.push(flow_main_session.input_name(i));
            }
        }
        let output_names = input_names.iter().map(|name| format!("out_{}", name)).collect();

        // initialize state tensors
        let mut name_to_index = HashMap::<String, usize>::new();
        for i in 0..input_count {
            name_to_index.insert(flow_main_session.input_name(i), i);
        }
        let mut states = Vec::<onnx::Value>::new();
        for name in &input_names {
            let index = match name_to_index.get(name) {
                Some(index) => *index,
                None => panic!("FlowMain: input name not found: {}", name),
            };
            let shape = flow_main_session.input_shape(index);
            let elem_type = flow_main_session.input_element_type(index);
            let tensor = if shape == [0] {
                onnx::Value::from_slice::<f32>(&onnx, &[0], &[])
            } else {
                match elem_type {
                    onnx::ONNXTensorElementDataType::Float => onnx::Value::zeros::<f32>(&onnx, &shape),
                    onnx::ONNXTensorElementDataType::Int64 => onnx::Value::zeros::<i64>(&onnx, &shape),
                    onnx::ONNXTensorElementDataType::Bool => {
                        let resolved: Vec<usize> = shape
                            .iter()
                            .map(|&value| if value < 0 { 1 } else { value as usize })
                            .collect();
                        let total: usize = resolved.iter().product();
                        let data = vec![true; total];
                        onnx::Value::from_slice::<bool>(&onnx, &resolved, &data)
                    }
                    other => panic!("FlowMain: unsupported element type: {:?}", other),
                }
            };
            states.push(tensor);
        }
        let sequence_tensor =
            onnx::Value::from_slice::<f32>(&flow_step_session.onnx, &[1, 1, LATENT_DIM], &vec![f32::NAN; LATENT_DIM]);
        let s_tensor = onnx::Value::zeros::<f32>(&flow_step_session.onnx, &[1, 1]);
        let t_tensor = onnx::Value::zeros::<f32>(&flow_step_session.onnx, &[1, 1]);
        let c_tensor = onnx::Value::zeros::<f32>(&flow_step_session.onnx, &[1, CONDITIONING_DIM as i64]);
        let x_tensor = onnx::Value::zeros::<f32>(&flow_step_session.onnx, &[1, LATENT_DIM as i64]);

        Self {
            flow_main_session,
            flow_step_session,
            conditioner_session,
            input_names,
            output_names,
            reset_state_tensors: states.iter().map(|v| v.deepclone()).collect::<Vec<_>>(),
            state_tensors: states,
            sequence_tensor,
            s_tensor,
            t_tensor,
            c_tensor,
            x_tensor,
        }
    }

    pub fn init_voice(&mut self, voice: &[f32]) {
        let sequence_tensor = onnx::Value::from_slice::<f32>(&self.flow_main_session.onnx, &[1, 0, LATENT_DIM], &[]);
        let text_embeddings_tensor = onnx::Value::from_slice::<f32>(
            &self.flow_main_session.onnx,
            &[1, voice.len() / CONDITIONING_DIM, CONDITIONING_DIM],
            &voice,
        );
        let mut inputs = vec![("sequence", &sequence_tensor), ("text_embeddings", &text_embeddings_tensor)];
        for (i, state_tensor) in self.state_tensors.iter().enumerate() {
            inputs.push((&self.input_names[i], state_tensor));
        }
        let output_names: Vec<&str> = self.output_names.iter().map(|s| s.as_str()).collect();
        self.reset_state_tensors = self.flow_main_session.run(&inputs, &output_names);
    }

    pub fn condition(&mut self, tokens: &[i64]) {
        let seq_len = tokens.len();
        let tokens_tensor = onnx::Value::from_slice::<i64>(&self.conditioner_session.onnx, &[1, seq_len], tokens);
        let outputs = self
            .conditioner_session
            .run(&[("token_ids", &tokens_tensor)], &["embeddings"]);
        let embeddings = outputs[0].extract_tensor::<f32>();
        let sequence_tensor = onnx::Value::from_slice::<f32>(&self.flow_main_session.onnx, &[1, 0, LATENT_DIM], &[]);
        let text_embeddings_tensor =
            onnx::Value::from_slice::<f32>(&self.flow_main_session.onnx, &[1, seq_len, CONDITIONING_DIM], &embeddings);
        let mut inputs = vec![("sequence", &sequence_tensor), ("text_embeddings", &text_embeddings_tensor)];
        for (i, state_tensor) in self.state_tensors.iter().enumerate() {
            inputs.push((&self.input_names[i], state_tensor));
        }
        let output_names: Vec<&str> = self.output_names.iter().map(|s| s.as_str()).collect();
        self.state_tensors = self.flow_main_session.run(&inputs, &output_names);
    }

    pub fn step(&mut self) -> (Vec<f32>, bool) {
        let text_embeddings_tensor =
            onnx::Value::from_slice::<f32>(&self.flow_main_session.onnx, &[1, 0, CONDITIONING_DIM], &[]);
        let mut inputs = vec![
            ("sequence", &self.sequence_tensor),
            ("text_embeddings", &text_embeddings_tensor),
        ];
        for (i, state_tensor) in self.state_tensors.iter().enumerate() {
            inputs.push((&self.input_names[i], state_tensor));
        }
        let mut output_names: Vec<&str> = vec!["conditioning", "eos_logit"];
        for name in &self.output_names {
            output_names.push(name.as_str());
        }
        let mut outputs = self.flow_main_session.run(&inputs, &output_names);
        let conditioning = outputs[0].extract_tensor::<f32>().to_vec();
        let eos_logit = outputs[1].extract_tensor::<f32>()[0];
        self.state_tensors = outputs.split_off(2);
        let mut rng = rand::thread_rng();
        let std = (DEFAULT_TEMPERATURE as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        let mut latent = [0f32; LATENT_DIM];
        for i in 0..LATENT_DIM {
            latent[i] = normal.sample(&mut rng) as f32;
        }
        for i in 0..DEFAULT_LSD_STEPS {
            self.s_tensor.as_slice_mut::<f32>()[0] = i as f32 / DEFAULT_LSD_STEPS as f32;
            self.t_tensor.as_slice_mut::<f32>()[0] = (i + 1) as f32 / DEFAULT_LSD_STEPS as f32;
            self.c_tensor.as_slice_mut::<f32>().copy_from_slice(&conditioning);
            self.x_tensor.as_slice_mut::<f32>().copy_from_slice(&latent);
            let outputs = self.flow_step_session.run(
                &[
                    ("c", &self.c_tensor),
                    ("s", &self.s_tensor),
                    ("t", &self.t_tensor),
                    ("x", &self.x_tensor),
                ],
                &["flow_dir"],
            );
            let flow_dir = outputs[0].extract_tensor::<f32>();
            for k in 0..LATENT_DIM {
                latent[k] += flow_dir[k] / DEFAULT_LSD_STEPS as f32;
            }
        }

        // store latent and prepare decoder input
        self.sequence_tensor.as_slice_mut::<f32>().copy_from_slice(&latent);

        // check if EOS
        (latent.to_vec(), eos_logit > DEFAULT_EOS_THRESHOLD)
    }

    pub fn reset(&mut self) {
        self.state_tensors = self.reset_state_tensors.iter().map(|v| v.deepclone()).collect::<Vec<_>>();
        self.sequence_tensor =
            onnx::Value::from_slice::<f32>(&self.flow_step_session.onnx, &[1, 1, LATENT_DIM], &vec![f32::NAN; LATENT_DIM]);
    }
}
