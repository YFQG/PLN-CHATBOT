import nltk
import numpy as np
import json
import os
from datetime import datetime
import re
import string

# Función segura para intentar importar bibliotecas
def importar_seguro(biblioteca, mensaje_alt=""):
    try:
        return __import__(biblioteca)
    except ImportError:
        print(f"No se pudo importar {biblioteca}. {mensaje_alt}")
        return None

# Intentar cargar spaCy (opcional)
spacy = importar_seguro("spacy", "El análisis será simplificado.")

# Intentar descargar recursos de NLTK de manera segura
try:
    print("Descargando recursos necesarios para NLTK...")
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print("Recursos descargados correctamente.")
except Exception as e:
    print(f"Error al descargar recursos NLTK: {e}")
    print("Se utilizarán funciones alternativas.")

# Funciones alternativas si las bibliotecas fallan
def tokenize_simple(texto):
    """Función simple de tokenización como alternativa a NLTK"""
    # Eliminar puntuación y convertir a minúsculas
    texto = texto.lower()
    # Eliminar puntuación
    texto = ''.join([char for char in texto if char not in string.punctuation])
    # Dividir por espacios
    return texto.split()

def pos_tag_simple(tokens):
    """Etiquetado POS simplificado cuando NLTK falla"""
    # Simplemente asumimos que todas las palabras son sustantivos (NN)
    return [(token, "NN") for token in tokens]

class AnalisisEmpresarialChat:
    def __init__(self):
        # Diccionario para almacenar datos de empresas
        self.empresas = {}
        self.cargar_datos()
        print("Sistema de Análisis Empresarial por Chat iniciado.")
        print("Escribe 'ayuda' para ver los comandos disponibles.")
        
        # Verificar si tenemos acceso a las funciones de NLTK
        self.use_nltk = True
        try:
            from nltk.tokenize import word_tokenize
            from nltk.tag import pos_tag
            self.word_tokenize = word_tokenize
            self.pos_tag = pos_tag
        except (ImportError, LookupError):
            print("NLTK no está disponible, se usarán funciones simplificadas.")
            self.word_tokenize = tokenize_simple
            self.pos_tag = pos_tag_simple
            self.use_nltk = False
        
        # Verificar si podemos usar spaCy
        self.use_spacy = False
        if spacy:
            try:
                self.nlp = spacy.load('es_core_news_md')
                self.use_spacy = True
                print("Modelo spaCy cargado correctamente.")
            except:
                try:
                    self.nlp = spacy.load('es_core_news_sm')
                    self.use_spacy = True
                    print("Usando modelo alternativo 'es_core_news_sm'")
                except:
                    print("No se pudo cargar ningún modelo de spaCy.")
    
    def cargar_datos(self):
        try:
            if os.path.exists("empresas_data.json"):
                with open("empresas_data.json", "r", encoding="utf-8") as file:
                    self.empresas = json.load(file)
                print(f"Se cargaron datos de {len(self.empresas)} empresas.")
            else:
                print("No se encontró archivo de datos. Se iniciará con una base de datos vacía.")
        except Exception as e:
            print(f"Error al cargar datos: {str(e)}")
            self.empresas = {}
    
    def guardar_datos(self):
        try:
            with open("empresas_data.json", "w", encoding="utf-8") as file:
                json.dump(self.empresas, file, ensure_ascii=False, indent=4)
            print("Datos guardados correctamente.")
        except Exception as e:
            print(f"Error al guardar datos: {str(e)}")
    
    def mostrar_ayuda(self):
        print("\n=== COMANDOS DISPONIBLES ===")
        print("ayuda - Muestra esta información")
        print("nueva empresa - Registra una nueva empresa")
        print("listar - Muestra las empresas registradas")
        print("analizar [nombre] - Analiza una empresa específica")
        print("buscar [término] - Busca empresas por nombre o sector")
        print("salir - Termina la aplicación")
        print("===========================\n")
    
    def registrar_empresa(self):
        print("\n=== REGISTRO DE NUEVA EMPRESA ===")
        datos = {}
        
        # Solicitar datos
        datos["nombre"] = input("Nombre de la Empresa: ")
        
        # Verificar si ya existe
        if datos["nombre"] in self.empresas:
            respuesta = input(f"La empresa '{datos['nombre']}' ya existe. ¿Desea actualizarla? (s/n): ")
            if respuesta.lower() != 's':
                print("Registro cancelado.")
                return
        
        try:
            datos["valor_anual"] = float(input("Valor Anual (COP): ").replace(',', '').replace('.', ''))
            datos["ganancias"] = float(input("Ganancias (COP): ").replace(',', '').replace('.', ''))
            datos["sector"] = input("Sector: ")
            datos["empleados"] = int(input("Número de Empleados: "))
            datos["activos"] = float(input("Valor en Activos (COP): ").replace(',', '').replace('.', ''))
            datos["cartera"] = float(input("Valor Cartera (COP): ").replace(',', '').replace('.', ''))
            datos["deudas"] = float(input("Valor Deudas (COP): ").replace(',', '').replace('.', ''))
            
            # Generar análisis
            print("\nGenerando análisis...")
            analisis = self.generar_analisis_nlp(
                datos["nombre"], datos["sector"], datos["valor_anual"], 
                datos["ganancias"], datos["empleados"], datos["activos"], 
                datos["cartera"], datos["deudas"]
            )
            
            # Mostrar análisis como texto
            self.mostrar_analisis_texto(datos["nombre"], datos["sector"], datos["valor_anual"], 
                                      datos["ganancias"], datos["empleados"], datos["activos"], 
                                      datos["cartera"], datos["deudas"], analisis)
            
            # Guardar datos completos
            datos["fecha_registro"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            datos["analisis_nlp"] = analisis
            
            # Almacenar en el diccionario
            self.empresas[datos["nombre"]] = datos
            self.guardar_datos()
            
            print(f"\nEmpresa '{datos['nombre']}' registrada correctamente.")
            print(f"Categoría financiera: {analisis['evaluacion']['categoria']}")
            
        except ValueError as e:
            print(f"Error: Por favor ingrese valores numéricos válidos. {str(e)}")
    
    def mostrar_analisis_texto(self, nombre, sector, valor_anual, ganancias, empleados, 
                           activos, cartera, deudas, analisis):
        """Muestra el análisis en formato texto"""
        resultado = f"""
ANÁLISIS FINANCIERO Y NLP DE {nombre.upper()}
=====================================

El análisis de la empresa {nombre}, perteneciente al sector {sector}, 
ha sido completado utilizando técnicas de procesamiento de lenguaje 
natural (NLP) y análisis financiero.

DATOS FINANCIEROS:
-----------------
• Valor anual: ${valor_anual:,.0f} COP
• Ganancias: ${ganancias:,.0f} COP
• Activos: ${activos:,.0f} COP
• Cartera: ${cartera:,.0f} COP
• Deudas: ${deudas:,.0f} COP
• Número de empleados: {empleados}

INDICADORES CALCULADOS:
---------------------
• Ratio de liquidez: {analisis['indicadores_financieros']['liquidez']:.2f}
• Margen de ganancia: {analisis['indicadores_financieros']['margen_ganancia']:.2f}%
• Ratio de endeudamiento: {analisis['indicadores_financieros']['ratio_endeudamiento']:.2f}%
• Productividad por empleado: ${analisis['indicadores_financieros']['productividad_empleado']:,.0f} COP

PROCESAMIENTO DE LENGUAJE:
------------------------
Se ha aplicado tokenización, lematización y etiquetado gramatical 
al nombre y sector de la empresa para su posterior análisis.

EVALUACIÓN GLOBAL:
----------------
La salud financiera de la empresa se clasifica como: {analisis['evaluacion']['categoria']}
Puntuación: {analisis['evaluacion']['puntuacion']}/100

DESCRIPCIÓN:
{analisis['evaluacion']['descripcion']}

RECOMENDACIONES:
--------------"""

        # Generar recomendaciones
        recomendaciones = []
        if analisis['indicadores_financieros']['liquidez'] < 1:
            recomendaciones.append("• Mejorar la posición de liquidez para cubrir obligaciones a corto plazo.")
        
        if analisis['indicadores_financieros']['margen_ganancia'] < 10:
            recomendaciones.append("• Implementar estrategias para aumentar el margen de ganancia.")
        
        if analisis['indicadores_financieros']['ratio_endeudamiento'] > 50:
            recomendaciones.append("• Reducir el nivel de endeudamiento para mejorar la estabilidad financiera.")
        
        if analisis['indicadores_financieros']['productividad_empleado'] < 100000000:
            recomendaciones.append("• Revisar la productividad por empleado para optimizar recursos.")
        
        # Si no hay recomendaciones específicas
        if not recomendaciones:
            recomendaciones.append("• La empresa muestra indicadores saludables. Se recomienda mantener las estrategias actuales.")
        
        for recomendacion in recomendaciones:
            resultado += f"\n{recomendacion}"
        
        print(resultado)
        return resultado
    
    def listar_empresas(self):
        if not self.empresas:
            print("No hay empresas registradas.")
            return
        
        print("\n=== EMPRESAS REGISTRADAS ===")
        print(f"{'NOMBRE':<30} {'SECTOR':<20} {'EMPLEADOS':<10} {'SALUD FINANCIERA':<20}")
        print("="*80)
        
        for nombre, datos in self.empresas.items():
            print(f"{datos['nombre']:<30} {datos['sector']:<20} {datos['empleados']:<10} {datos['analisis_nlp']['evaluacion']['categoria']:<20}")
    
    def buscar_empresas(self, termino):
        if not self.empresas:
            print("No hay empresas registradas.")
            return
        
        termino = termino.lower()
        resultados = []
        
        for nombre, datos in self.empresas.items():
            if termino in nombre.lower() or termino in datos['sector'].lower():
                resultados.append(datos)
        
        if not resultados:
            print(f"No se encontraron empresas con el término '{termino}'.")
            return
        
        print(f"\n=== RESULTADOS DE BÚSQUEDA PARA '{termino}' ===")
        print(f"{'NOMBRE':<30} {'SECTOR':<20} {'EMPLEADOS':<10} {'SALUD FINANCIERA':<20}")
        print("="*80)
        
        for datos in resultados:
            print(f"{datos['nombre']:<30} {datos['sector']:<20} {datos['empleados']:<10} {datos['analisis_nlp']['evaluacion']['categoria']:<20}")
    
    def analizar_empresa(self, nombre):
        if not nombre:
            print("Por favor especifique el nombre de la empresa.")
            return
            
        if nombre not in self.empresas:
            print(f"No se encontró la empresa '{nombre}'.")
            sugerencias = [n for n in self.empresas.keys() if nombre.lower() in n.lower()]
            if sugerencias:
                print("Quizás quiso decir:")
                for sugerencia in sugerencias:
                    print(f"- {sugerencia}")
            return
        
        datos = self.empresas[nombre]
        self.mostrar_analisis_texto(
            datos["nombre"], datos["sector"], datos["valor_anual"], 
            datos["ganancias"], datos["empleados"], datos["activos"],
            datos["cartera"], datos["deudas"], datos["analisis_nlp"]
        )
    
    def generar_analisis_nlp(self, nombre, sector, valor_anual, ganancias, 
                            empleados, activos, cartera, deudas):
        resultado = {}
        
        # 1. Tokenización del nombre de la empresa y sector
        try:
            tokens_nombre = self.word_tokenize(nombre.lower())
            tokens_sector = self.word_tokenize(sector.lower())
        except Exception as e:
            print(f"Error en tokenización: {e}")
            tokens_nombre = nombre.lower().split()
            tokens_sector = sector.lower().split()
        
        resultado["tokenizacion"] = {
            "nombre": tokens_nombre,
            "sector": tokens_sector
        }
        
        # 2. Lematización (simplificada si no hay NLTK)
        if self.use_nltk:
            try:
                from nltk.stem import WordNetLemmatizer
                lemmatizer = WordNetLemmatizer()
                lemmas_nombre = [lemmatizer.lemmatize(token) for token in tokens_nombre]
                lemmas_sector = [lemmatizer.lemmatize(token) for token in tokens_sector]
            except:
                lemmas_nombre = tokens_nombre
                lemmas_sector = tokens_sector
        else:
            # Sin NLTK, solo usamos los tokens como lemas
            lemmas_nombre = tokens_nombre
            lemmas_sector = tokens_sector
        
        resultado["lematizacion"] = {
            "nombre": lemmas_nombre,
            "sector": lemmas_sector
        }
        
        # 3. POS Tagging (Etiquetado de partes del discurso)
        try:
            pos_nombre = self.pos_tag(tokens_nombre)
            pos_sector = self.pos_tag(tokens_sector)
        except Exception as e:
            print(f"Error en etiquetado POS: {e}")
            pos_nombre = [(token, "UNK") for token in tokens_nombre]
            pos_sector = [(token, "UNK") for token in tokens_sector]
        
        resultado["pos_tagging"] = {
            "nombre": pos_nombre,
            "sector": pos_sector
        }
        
        # 4. Embeddings utilizando spaCy (si está disponible)
        if self.use_spacy:
            try:
                doc_nombre = self.nlp(" ".join(tokens_nombre))
                doc_sector = self.nlp(" ".join(tokens_sector))
                # Guardar solo los valores del vector para facilitar la serialización
                resultado["embeddings"] = {
                    "nombre": doc_nombre.vector.tolist(),
                    "sector": doc_sector.vector.tolist()
                }
            except Exception as e:
                print(f"Error al generar embeddings: {e}")
                resultado["embeddings"] = {
                    "nombre": [0] * 10,  # Vector vacío como fallback
                    "sector": [0] * 10
                }
        else:
            # Sin spaCy, creamos embeddings ficticios
            resultado["embeddings"] = {
                "nombre": [0] * 10,
                "sector": [0] * 10
            }
        
        # 5. Análisis financiero para categorización
        # Calcular indicadores financieros
        liquidez = activos / deudas if deudas > 0 else float('inf')
        margen_ganancia = (ganancias / valor_anual) * 100 if valor_anual > 0 else 0
        ratio_endeudamiento = (deudas / activos) * 100 if activos > 0 else float('inf')
        productividad_empleado = valor_anual / empleados if empleados > 0 else 0
        
        resultado["indicadores_financieros"] = {
            "liquidez": liquidez,
            "margen_ganancia": margen_ganancia,
            "ratio_endeudamiento": ratio_endeudamiento,
            "productividad_empleado": productividad_empleado
        }
        
        # Evaluación de la salud financiera
        puntuacion = 0
        max_puntuacion = 100
        
        # Evaluar liquidez (25%)
        if liquidez >= 2:
            puntuacion += 25
        elif liquidez >= 1.5:
            puntuacion += 20
        elif liquidez >= 1:
            puntuacion += 15
        elif liquidez >= 0.5:
            puntuacion += 10
        else:
            puntuacion += 5
        
        # Evaluar margen de ganancia (25%)
        if margen_ganancia >= 20:
            puntuacion += 25
        elif margen_ganancia >= 15:
            puntuacion += 20
        elif margen_ganancia >= 10:
            puntuacion += 15
        elif margen_ganancia >= 5:
            puntuacion += 10
        else:
            puntuacion += 5
        
        # Evaluar endeudamiento (25%)
        if ratio_endeudamiento <= 30:
            puntuacion += 25
        elif ratio_endeudamiento <= 40:
            puntuacion += 20
        elif ratio_endeudamiento <= 50:
            puntuacion += 15
        elif ratio_endeudamiento <= 60:
            puntuacion += 10
        else:
            puntuacion += 5
        
        # Evaluar productividad (25%)
        if productividad_empleado >= 200000000:  # 200 millones por empleado
            puntuacion += 25
        elif productividad_empleado >= 150000000:
            puntuacion += 20
        elif productividad_empleado >= 100000000:
            puntuacion += 15
        elif productividad_empleado >= 50000000:
            puntuacion += 10
        else:
            puntuacion += 5
        
        # Categorizar la salud financiera
        if puntuacion >= 85:
            categoria = "Excelente"
            descripcion = "La empresa muestra una salud financiera excepcional."
        elif puntuacion >= 70:
            categoria = "Muy Buena"
            descripcion = "La empresa tiene una posición financiera sólida."
        elif puntuacion >= 55:
            categoria = "Buena"
            descripcion = "La empresa presenta indicadores financieros estables."
        elif puntuacion >= 40:
            categoria = "Regular"
            descripcion = "La empresa tiene áreas que necesitan mejoras."
        elif puntuacion >= 25:
            categoria = "Deficiente"
            descripcion = "La empresa presenta problemas financieros significativos."
        else:
            categoria = "Crítica"
            descripcion = "La empresa requiere atención urgente en su gestión financiera."
        
        resultado["evaluacion"] = {
            "puntuacion": puntuacion,
            "max_puntuacion": max_puntuacion,
            "categoria": categoria,
            "descripcion": descripcion
        }
        
        return resultado
    
    def analizar_texto(self, pregunta):
        """Analiza preguntas en lenguaje natural y responde según el contexto"""
        pregunta = pregunta.lower()
        
        # Verificar si es una pregunta sobre empresas específicas
        for nombre in self.empresas.keys():
            if nombre.lower() in pregunta:
                if "indicadores" in pregunta or "financi" in pregunta:
                    datos = self.empresas[nombre]
                    print(f"\nIndicadores financieros de {nombre}:")
                    print(f"• Liquidez: {datos['analisis_nlp']['indicadores_financieros']['liquidez']:.2f}")
                    print(f"• Margen de ganancia: {datos['analisis_nlp']['indicadores_financieros']['margen_ganancia']:.2f}%")
                    print(f"• Ratio de endeudamiento: {datos['analisis_nlp']['indicadores_financieros']['ratio_endeudamiento']:.2f}%")
                    return True
                elif "recomend" in pregunta:
                    self.mostrar_recomendaciones(nombre)
                    return True
                else:
                    self.analizar_empresa(nombre)
                    return True
        
        # Preguntas generales sobre todas las empresas
        if "mejor empresa" in pregunta or "empresa con mejor" in pregunta:
            self.mostrar_mejor_empresa()
            return True
        elif "peor empresa" in pregunta or "empresa con peor" in pregunta:
            self.mostrar_peor_empresa()
            return True
        elif "cuántas empresas" in pregunta or "número de empresas" in pregunta:
            print(f"Hay {len(self.empresas)} empresas registradas en el sistema.")
            return True
        elif "sectores" in pregunta:
            self.mostrar_sectores()
            return True
        
        # No se encontró una pregunta específica
        return False
    
    def mostrar_recomendaciones(self, nombre):
        if nombre not in self.empresas:
            print(f"No se encontró la empresa '{nombre}'.")
            return
            
        datos = self.empresas[nombre]
        analisis = datos["analisis_nlp"]
        
        print(f"\nRECOMENDACIONES PARA {nombre.upper()}:")
        
        # Generar recomendaciones basadas en los indicadores
        recomendaciones = []
        
        if analisis['indicadores_financieros']['liquidez'] < 1:
            recomendaciones.append("• Se recomienda mejorar la posición de liquidez para cubrir obligaciones a corto plazo.")
        
        if analisis['indicadores_financieros']['margen_ganancia'] < 10:
            recomendaciones.append("• Se recomienda implementar estrategias para aumentar el margen de ganancia.")
        
        if analisis['indicadores_financieros']['ratio_endeudamiento'] > 50:
            recomendaciones.append("• Se recomienda reducir el nivel de endeudamiento para mejorar la estabilidad financiera.")
        
        if analisis['indicadores_financieros']['productividad_empleado'] < 100000000:
            recomendaciones.append("• Se recomienda revisar la productividad por empleado para optimizar recursos.")
        
        # Si no hay recomendaciones específicas
        if not recomendaciones:
            recomendaciones.append("• La empresa muestra indicadores saludables. Se recomienda mantener las estrategias actuales.")
        
        for recomendacion in recomendaciones:
            print(recomendacion)
    
    def mostrar_mejor_empresa(self):
        if not self.empresas:
            print("No hay empresas registradas.")
            return
            
        mejor_empresa = None
        mejor_puntuacion = -1
        
        for nombre, datos in self.empresas.items():
            puntuacion = datos["analisis_nlp"]["evaluacion"]["puntuacion"]
            if puntuacion > mejor_puntuacion:
                mejor_puntuacion = puntuacion
                mejor_empresa = datos
        
        print(f"\nLa empresa con mejor salud financiera es: {mejor_empresa['nombre']}")
        print(f"Sector: {mejor_empresa['sector']}")
        print(f"Puntuación: {mejor_empresa['analisis_nlp']['evaluacion']['puntuacion']}/100")
        print(f"Categoría: {mejor_empresa['analisis_nlp']['evaluacion']['categoria']}")
    
    def mostrar_peor_empresa(self):
        if not self.empresas:
            print("No hay empresas registradas.")
            return
            
        peor_empresa = None
        peor_puntuacion = float('inf')
        
        for nombre, datos in self.empresas.items():
            puntuacion = datos["analisis_nlp"]["evaluacion"]["puntuacion"]
            if puntuacion < peor_puntuacion:
                peor_puntuacion = puntuacion
                peor_empresa = datos
        
        print(f"\nLa empresa con salud financiera más baja es: {peor_empresa['nombre']}")
        print(f"Sector: {peor_empresa['sector']}")
        print(f"Puntuación: {peor_empresa['analisis_nlp']['evaluacion']['puntuacion']}/100")
        print(f"Categoría: {peor_empresa['analisis_nlp']['evaluacion']['categoria']}")
    
    def mostrar_sectores(self):
        if not self.empresas:
            print("No hay empresas registradas.")
            return
            
        sectores = {}
        for nombre, datos in self.empresas.items():
            sector = datos["sector"]
            if sector in sectores:
                sectores[sector] += 1
            else:
                sectores[sector] = 1
        
        print("\nSECTORES REGISTRADOS:")
        for sector, cantidad in sectores.items():
            print(f"• {sector}: {cantidad} empresas")
    
    def iniciar_chat(self):
        """Inicia el loop principal del chat"""
        while True:
            print("\n> ", end="")
            entrada = input().strip()
            
            if entrada.lower() == "salir":
                print("¡Hasta pronto!")
                break
            elif entrada.lower() == "ayuda":
                self.mostrar_ayuda()
            elif entrada.lower() == "nueva empresa":
                self.registrar_empresa()
            elif entrada.lower() == "listar":
                self.listar_empresas()
            elif entrada.lower().startswith("buscar "):
                termino = entrada[7:].strip()
                self.buscar_empresas(termino)
            elif entrada.lower().startswith("analizar "):
                nombre = entrada[9:].strip()
                self.analizar_empresa(nombre)
            else:
                # Intentar interpretar la pregunta en lenguaje natural
                if not self.analizar_texto(entrada):
                    print("No entiendo esa petición. Escribe 'ayuda' para ver los comandos disponibles.")

if __name__ == "__main__":
    sistema = AnalisisEmpresarialChat()
    sistema.iniciar_chat()