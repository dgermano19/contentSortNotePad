from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import uuid

app = Flask(__name__)

# Inicializando o pipeline de NLP da Hugging Face para categorização
# Altere "model_name" para o modelo que você deseja usar.
nlp_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Dicionário para armazenar as notas em memória
notes = {}

# Possíveis categorias para simulação
categories = ["Trabalho", "Estudo", "Pessoal", "Urgente", "Lazer", "Projetos"]

# Endpoint para criar uma nova nota
@app.route("/notes", methods=["POST"])
def create_note():
    data = request.json
    note_id = str(uuid.uuid4())
    notes[note_id] = {
        "title": data.get("title"),
        "content": data.get("content"),
        "category": "Não categorizada"
    }
    return jsonify({"id": note_id}), 201

# Endpoint para obter todas as notas
@app.route("/notes", methods=["GET"])
def get_notes():
    return jsonify(notes), 200

# Endpoint para obter ou atualizar uma nota específica
@app.route("/notes/<note_id>", methods=["GET", "PUT"])
def note_detail(note_id):
    if request.method == "GET":
        note = notes.get(note_id)
        if note is None:
            return jsonify({"error": "Nota não encontrada"}), 404
        return jsonify(note), 200

    elif request.method == "PUT":
        data = request.json
        note = notes.get(note_id)
        if note is None:
            return jsonify({"error": "Nota não encontrada"}), 404
        note["title"] = data.get("title", note["title"])
        note["content"] = data.get("content", note["content"])
        return jsonify(note), 200

# Endpoint para excluir uma nota
@app.route("/notes/<note_id>", methods=["DELETE"])
def delete_note(note_id):
    if note_id in notes:
        del notes[note_id]
        return jsonify({"message": "Nota excluída com sucesso"}), 200
    return jsonify({"error": "Nota não encontrada"}), 404

# Endpoint para organizar as notas usando o modelo NLP
@app.route("/organize", methods=["POST"])
def organize_notes():
    organized_notes = {}
    for note_id, note in notes.items():
        # Usando o modelo NLP para sugerir uma categoria
        result = nlp_model(note["content"], candidate_labels=categories)
        suggested_category = result["labels"][0]  # A primeira categoria sugerida
        organized_notes[note_id] = {
            "title": note["title"],
            "content": note["content"],
            "suggested_category": suggested_category
        }
    return jsonify(organized_notes), 200

# Endpoint para aplicar a categoria sugerida a uma nota específica
@app.route("/apply_category/<note_id>", methods=["PUT"])
def apply_category(note_id):
    data = request.json
    note = notes.get(note_id)
    if note is None:
        return jsonify({"error": "Nota não encontrada"}), 404
    note["category"] = data.get("suggested_category", note["category"])
    return jsonify({"message": "Categoria aplicada com sucesso", "note": note}), 200


@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
