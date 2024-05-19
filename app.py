from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

combinations = {}

@app.route('/combine', methods=['POST'])
def combine():
    data = request.json
    element1 = data['element1']
    element2 = data['element2']
    
    combination_key = f"{element1}+{element2}"
    
    if combination_key in combinations:
        result = combinations[combination_key]
    else:
        prompt = f"Combine {element1} and {element2}"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs['input_ids'], max_length=50)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        combinations[combination_key] = result
    
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
