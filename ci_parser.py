# ci_parser_fuzzy_clean.py

import re
import json
from fuzzywuzzy import fuzz

class CIParser:
    def __init__(self, threshold=70):
        """
        threshold: porcentaje mínimo de similitud para considerar un label como coincidencia
        """
        self.threshold = threshold
        # Variantes de labels que queremos detectar
        self.label_map = {
            "apellidos": ["apellidos", "apellido", "apellido y nombre", "apellidos y nombres", "apellido/s y nombre/s"],
            "fecha_nacimiento": ["fecha de nacimiento", "nacimiento", "f.nac"],
            "lugar_nacimiento": ["lugar de nacimiento", "nacido en", "lugar de naci"],
            "fecha_vencimiento": ["fecha de vencimiento", "vence", "vencimiento"],
            "sexo": ["sexo", "genero", "género", "sex"],
            "numero_documento": ["documento", "num. documento", "número de documento", "dni", "cedula"]
        }

    # -----------------------------
    # Limpieza de OCR
    # -----------------------------
    def _clean_ocr_text(self, ocr_text: str) -> str:
        """
        Limpia texto OCR eliminando líneas vacías y espacios innecesarios.
        """
        # Dividir en líneas
        lines = ocr_text.splitlines()
        # Quitar espacios en blanco al inicio/final y eliminar líneas vacías
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        # Unir nuevamente en un solo texto
        return "\n".join(cleaned_lines)

    # -----------------------------
    # Matching de labels
    # -----------------------------
    def _match_label(self, text):
        """
        Devuelve la clave del campo con mayor similitud usando fuzzywuzzy
        """
        best_score = 0
        best_key = None
        for key, labels in self.label_map.items():
            for label in labels:
                score = fuzz.ratio(text.lower(), label.lower())
                if score > best_score:
                    best_score = score
                    best_key = key
        if best_score >= self.threshold:
            return best_key
        return None

    # -----------------------------
    # Parsing principal
    # -----------------------------
    def parse(self, ocr_text: str) -> dict:
        """
        Recibe texto OCR, limpia y devuelve datos estructurados
        """
        self.fields = {
            "apellidos": "",
            "nombres": "",
            "fecha_nacimiento": "",
            "lugar_nacimiento": "",
            "fecha_vencimiento": "",
            "sexo": "",
            "numero_documento": ""
        }

        cleaned_text = self._clean_ocr_text(ocr_text)
        lines = [l.strip() for l in cleaned_text.split("\n") if l.strip()]

        skip_next = False
        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue

            label_key = self._match_label(line)
            if label_key == "apellidos":
                # Tomar la siguiente línea como apellidos
                if i + 1 < len(lines):
                    self.fields["apellidos"] = lines[i+1]
                # La siguiente línea después de los apellidos como nombres
                if i + 2 < len(lines):
                    next_label = self._match_label(lines[i+2])
                    if next_label is None:  # Si no es otra etiqueta
                        self.fields["nombres"] = lines[i+2]
                        skip_next = True  # Saltar esta línea en la siguiente iteración
            elif label_key in ["fecha_nacimiento", "lugar_nacimiento", "fecha_vencimiento", "sexo"]:
                if i + 1 < len(lines):
                    self.fields[label_key] = lines[i+1]
            elif label_key == "numero_documento":
                self.fields["numero_documento"] = line

        # Fallback para número de documento si no se detectó
        if not self.fields["numero_documento"]:
            match = re.search(r"\b\d{6,8}\b", cleaned_text)
            if match:
                self.fields["numero_documento"] = match.group()

        return self.fields

    # -----------------------------
    # Salida JSON
    # -----------------------------
    def to_json(self) -> str:
        return json.dumps(self.fields, indent=2, ensure_ascii=False)
