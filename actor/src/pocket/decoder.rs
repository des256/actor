use {
    super::*,
    std::{collections::HashMap, sync::Arc},
};
pub struct Decoder {
    session: onnx::Session,
    input_names: Vec<String>,
    output_names: Vec<String>,
    states: Vec<onnx::Value>,
}

impl Decoder {
    pub fn new(onnx: &Arc<onnx::Onnx>, executor: onnx::Executor) -> Self {
        let session = onnx.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, DECODER_PATH);

        // extract input and output names
        let mut input_names = Vec::<String>::new();
        let input_count = session.input_count();
        for i in 0..input_count {
            let name = session.input_name(i);
            if !["latent"].contains(&name.as_str()) {
                input_names.push(name);
            }
        }
        let output_names = input_names.iter().map(|name| format!("out_{}", name)).collect();

        // initialize state tensors
        let mut name_to_index = HashMap::<String, usize>::new();
        for i in 0..input_count {
            name_to_index.insert(session.input_name(i), i);
        }
        let mut states = Vec::<onnx::Value>::new();
        for name in &input_names {
            let index = match name_to_index.get(name) {
                Some(index) => *index,
                None => panic!("FlowMain: input name not found: {}", name),
            };
            let shape = session.input_shape(index);
            let elem_type = session.input_element_type(index);
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

        Self {
            session,
            input_names,
            output_names,
            states,
        }
    }

    pub fn decode(&mut self, latent: &[f32]) -> Vec<i16> {
        let latent_tensor = onnx::Value::from_slice::<f32>(&self.session.onnx, &[1, 1, latent.len()], &latent);
        let mut inputs = vec![("latent", &latent_tensor)];
        for (i, state) in self.states.iter().enumerate() {
            inputs.push((&self.input_names[i], state));
        }
        let mut output_names = vec!["audio_frame"];
        output_names.extend(self.output_names.iter().map(|name| name.as_str()));
        let mut outputs = self.session.run(&inputs, &output_names);
        self.states = outputs.split_off(1);
        outputs[0]
            .extract_tensor::<f32>()
            .iter()
            .map(|&value| (value * 32768.0).clamp(-32768.0, 32767.0) as i16)
            .collect()
    }
}
