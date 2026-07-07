class KANFeatureExtractor:
    def __init__(self, width, grid, k):
        # Los hiperparámetros y pesos ocupan un espacio despreciable en RAM
        self.width = width
        self.grid = grid
        self.k = k
        self.model = None  # Se inicializará en el fit
        self.symbolic_formulas = [] # Ocupa bytes, es seguro guardarlo

    def fit_and_prune(self, X_train, y_train, X_val, y_val):
        # 1. El tensor entra como argumento (no se guarda en self)
        self.model = KAN(self.width, self.grid, self.k)
        
        # ... Lógica de entrenamiento y poda ...
        
        # 2. Al terminar el entrenamiento, extraes lo que te interesa
        self.symbolic_formulas = self._extract_formulas()
        
        # 3. Liberas el modelo completo si ya no lo necesitas en RAM
        del self.model
        self.model = None 
        # Aquí los tensores de entrada pierden su referencia local y pueden ser liberados por gc
