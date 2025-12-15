"""
Interface graphique simplifiée pour le projet Knapsack
"""

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class KnapsackGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Knapsack - Comparaison d'Algorithmes")
        self.root.geometry("1100x700")
        self.root.configure(bg='white')
        
        self.current_image_index = 0
        self.graph_images = []
        self.photo_images = []
        self.benchmark_running = False
        
        self.create_widgets()
        self.load_existing_graphs()
        
    def create_widgets(self):
        """Interface simplifiée"""
        
        # === TITRE ===
        title = tk.Label(self.root, text="Projet Knapsack", 
                        font=('Arial', 24, 'bold'), bg='white', fg='#333')
        title.pack(pady=15)
        
        # === SELECTION TAILLE ===
        size_frame = tk.Frame(self.root, bg='white')
        size_frame.pack(pady=5)
        
        tk.Label(size_frame, text="Taille:", font=('Arial', 11), bg='white', fg='#333').pack(side=tk.LEFT, padx=5)
        
        self.size_vars = {}
        size_labels = {'tiny': 'Tiny (n<=20)', 'small': 'Small', 'medium': 'Medium', 'large': 'Large', 'generated': 'Generated'}
        for size in ['tiny', 'small', 'medium', 'large', 'generated']:
            var = tk.BooleanVar(value=(size in ['tiny', 'small', 'medium']))
            self.size_vars[size] = var
            cb = tk.Checkbutton(size_frame, text=size_labels[size], variable=var,
                               font=('Arial', 10), bg='white', activebackground='white')
            cb.pack(side=tk.LEFT, padx=8)
        
        # === BOUTONS ===
        btn_frame = tk.Frame(self.root, bg='white')
        btn_frame.pack(pady=10)
        
        self.btn_benchmark = tk.Button(btn_frame, text="Lancer Benchmark", 
                                       command=self.run_benchmark,
                                       font=('Arial', 12), bg='#3498db', fg='white',
                                       padx=20, pady=10, cursor='hand2', relief=tk.FLAT)
        self.btn_benchmark.pack(side=tk.LEFT, padx=10)
        
        self.btn_generate = tk.Button(btn_frame, text="Generer Graphiques", 
                                      command=self.generate_graphs,
                                      font=('Arial', 12), bg='#3498db', fg='white',
                                      padx=20, pady=10, cursor='hand2', relief=tk.FLAT)
        self.btn_generate.pack(side=tk.LEFT, padx=10)
        
        self.btn_gen_instances = tk.Button(btn_frame, text="Generer Instances", 
                                           command=self.open_instance_generator,
                                           font=('Arial', 12), bg='#27ae60', fg='white',
                                           padx=20, pady=10, cursor='hand2', relief=tk.FLAT)
        self.btn_gen_instances.pack(side=tk.LEFT, padx=10)
        
        # Status
        self.status = tk.Label(self.root, text="", font=('Arial', 10), bg='white', fg='#666')
        self.status.pack(pady=5)
        
        # === ZONE GRAPHIQUE ===
        graph_frame = tk.Frame(self.root, bg='white')
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Navigation gauche
        self.btn_prev = tk.Button(graph_frame, text="<", command=self.prev_graph,
                                 font=('Arial', 20), bg='#eee', fg='#333',
                                 width=3, cursor='hand2', relief=tk.FLAT)
        self.btn_prev.pack(side=tk.LEFT, padx=5)
        
        # Canvas pour l'image
        self.canvas = tk.Canvas(graph_frame, bg='#f5f5f5', highlightthickness=1,
                               highlightbackground='#ddd')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Navigation droite
        self.btn_next = tk.Button(graph_frame, text=">", command=self.next_graph,
                                 font=('Arial', 20), bg='#eee', fg='#333',
                                 width=3, cursor='hand2', relief=tk.FLAT)
        self.btn_next.pack(side=tk.RIGHT, padx=5)
        
        # === TITRE GRAPHIQUE + INDICATEUR ===
        bottom_frame = tk.Frame(self.root, bg='white')
        bottom_frame.pack(pady=10)
        
        self.graph_title = tk.Label(bottom_frame, text="", font=('Arial', 12), 
                                   bg='white', fg='#333')
        self.graph_title.pack()
        
        self.indicator = tk.Label(bottom_frame, text="", font=('Arial', 10), 
                                 bg='white', fg='#999')
        self.indicator.pack()
        
    def load_existing_graphs(self):
        """Charge les graphiques existants au demarrage"""
        graphs_dir = os.path.join(os.path.dirname(__file__), 'results', 'graphs')
        
        if os.path.exists(graphs_dir):
            self.graph_images = sorted([f for f in os.listdir(graphs_dir) if f.endswith('.png')])
            if self.graph_images:
                self.status.config(text=f"{len(self.graph_images)} graphiques disponibles")
                self.root.after(100, lambda: self.display_graph(0))
            else:
                self.show_placeholder()
        else:
            self.show_placeholder()
            
    def show_placeholder(self):
        """Affiche un message par defaut"""
        self.canvas.delete("all")
        self.canvas.create_text(400, 250, text="Cliquez sur 'Generer Graphiques'\npour afficher les resultats",
                               font=('Arial', 14), fill='#999', justify=tk.CENTER)
        
    def run_benchmark(self):
        """Lance le benchmark"""
        if self.benchmark_running:
            return
        
        # Récupérer les tailles sélectionnées
        selected = [size for size, var in self.size_vars.items() if var.get()]
        if not selected:
            messagebox.showwarning("Attention", "Selectionnez au moins une taille")
            return
            
        self.benchmark_running = True
        self.btn_benchmark.config(state=tk.DISABLED, text="En cours...", bg='#888')
        self.status.config(text="Benchmark en cours... (peut prendre quelques minutes)")
        
        thread = threading.Thread(target=lambda: self._run_benchmark_thread(selected))
        thread.daemon = True
        thread.start()
        
    def _run_benchmark_thread(self, categories):
        """Execute le benchmark"""
        try:
            from src.benchmark import KnapsackBenchmark
            benchmark = KnapsackBenchmark()
            benchmark.run_benchmark(categories=categories)
            benchmark.save_results()
            self.root.after(0, lambda: self._benchmark_done(True))
        except Exception as e:
            self.root.after(0, lambda: self._benchmark_done(False, str(e)))
            
    def _benchmark_done(self, success, error=None):
        """Callback fin benchmark"""
        self.benchmark_running = False
        self.btn_benchmark.config(state=tk.NORMAL, text="Lancer Benchmark", bg='#3498db')
        
        if success:
            self.status.config(text="Benchmark termine ! Cliquez sur 'Generer Graphiques'")
            messagebox.showinfo("Succes", "Benchmark termine !")
        else:
            self.status.config(text=f"Erreur: {error}")
            
    def generate_graphs(self):
        """Genere les graphiques"""
        self.btn_generate.config(state=tk.DISABLED, text="Generation...", bg='#888')
        self.status.config(text="Generation des graphiques...")
        
        thread = threading.Thread(target=self._generate_thread)
        thread.daemon = True
        thread.start()
        
    def _generate_thread(self):
        """Genere les graphiques"""
        try:
            from src.visualizations import generate_all_graphs
            generate_all_graphs()
            self.root.after(0, self._graphs_done)
        except Exception as e:
            self.root.after(0, lambda: self._graphs_error(str(e)))
            
    def _graphs_done(self):
        """Callback fin generation"""
        self.btn_generate.config(state=tk.NORMAL, text="Generer Graphiques", bg='#2196F3')
        self.load_existing_graphs()
        self.status.config(text=f"{len(self.graph_images)} graphiques generes")
        
    def _graphs_error(self, error):
        """Callback erreur"""
        self.btn_generate.config(state=tk.NORMAL, text="Generer Graphiques", bg='#2196F3')
        self.status.config(text=f"Erreur: {error}")
        
    def display_graph(self, index):
        """Affiche un graphique"""
        if not self.graph_images or index >= len(self.graph_images):
            return
            
        self.current_image_index = index
        graphs_dir = os.path.join(os.path.dirname(__file__), 'results', 'graphs')
        img_path = os.path.join(graphs_dir, self.graph_images[index])
        
        try:
            img = Image.open(img_path)
            
            # Adapter a la taille du canvas
            self.canvas.update()
            cw = self.canvas.winfo_width() or 800
            ch = self.canvas.winfo_height() or 500
            
            ratio = min(cw / img.width, ch / img.height) * 0.95
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.photo_images = [photo]
            
            self.canvas.delete("all")
            self.canvas.create_image(cw//2, ch//2, image=photo, anchor=tk.CENTER)
            
            # Titre
            names = {
                'graph1_temps_execution.png': 'Temps d\'execution',
                'graph2_qualite_solutions.png': 'Qualite des solutions',
                'graph3_explosion_exponentielle.png': 'Explosion exponentielle',
                'graph4_grande_instance.png': 'Grande instance (n=500)',
                'graph5_tableau_synthese.png': 'Tableau de synthese',
                'graph6_recommandations.png': 'Recommandations',
            }
            name = names.get(self.graph_images[index], self.graph_images[index])
            self.graph_title.config(text=name)
            self.indicator.config(text=f"{index + 1} / {len(self.graph_images)}")
            
        except Exception as e:
            self.status.config(text=f"Erreur: {str(e)}")
            
    def prev_graph(self):
        """Graphique precedent"""
        if self.graph_images and self.current_image_index > 0:
            self.display_graph(self.current_image_index - 1)
            
    def next_graph(self):
        """Graphique suivant"""
        if self.graph_images and self.current_image_index < len(self.graph_images) - 1:
            self.display_graph(self.current_image_index + 1)
    
    def open_instance_generator(self):
        """Ouvre une fenetre pour generer des instances"""
        gen_window = tk.Toplevel(self.root)
        gen_window.title("Generateur d'Instances")
        gen_window.geometry("400x350")
        gen_window.configure(bg='white')
        gen_window.transient(self.root)
        gen_window.grab_set()
        
        # Centrer
        gen_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 400) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 350) // 2
        gen_window.geometry(f"+{x}+{y}")
        
        tk.Label(gen_window, text="Generateur d'Instances", 
                font=('Arial', 16, 'bold'), bg='white').pack(pady=15)
        
        # Nombre d'objets
        f1 = tk.Frame(gen_window, bg='white')
        f1.pack(pady=8, padx=20, fill=tk.X)
        tk.Label(f1, text="Nombre d'objets (n):", font=('Arial', 11), bg='white', width=20, anchor='w').pack(side=tk.LEFT)
        self.gen_n = tk.Entry(f1, font=('Arial', 11), width=10)
        self.gen_n.insert(0, "50")
        self.gen_n.pack(side=tk.LEFT, padx=5)
        
        # Capacité
        f2 = tk.Frame(gen_window, bg='white')
        f2.pack(pady=8, padx=20, fill=tk.X)
        tk.Label(f2, text="Capacite (W):", font=('Arial', 11), bg='white', width=20, anchor='w').pack(side=tk.LEFT)
        self.gen_w = tk.Entry(f2, font=('Arial', 11), width=10)
        self.gen_w.insert(0, "1000")
        self.gen_w.pack(side=tk.LEFT, padx=5)
        
        # Poids max
        f3 = tk.Frame(gen_window, bg='white')
        f3.pack(pady=8, padx=20, fill=tk.X)
        tk.Label(f3, text="Poids max:", font=('Arial', 11), bg='white', width=20, anchor='w').pack(side=tk.LEFT)
        self.gen_wmax = tk.Entry(f3, font=('Arial', 11), width=10)
        self.gen_wmax.insert(0, "100")
        self.gen_wmax.pack(side=tk.LEFT, padx=5)
        
        # Valeur max
        f4 = tk.Frame(gen_window, bg='white')
        f4.pack(pady=8, padx=20, fill=tk.X)
        tk.Label(f4, text="Valeur max:", font=('Arial', 11), bg='white', width=20, anchor='w').pack(side=tk.LEFT)
        self.gen_vmax = tk.Entry(f4, font=('Arial', 11), width=10)
        self.gen_vmax.insert(0, "200")
        self.gen_vmax.pack(side=tk.LEFT, padx=5)
        
        # Nom du fichier
        f5 = tk.Frame(gen_window, bg='white')
        f5.pack(pady=8, padx=20, fill=tk.X)
        tk.Label(f5, text="Nom du fichier:", font=('Arial', 11), bg='white', width=20, anchor='w').pack(side=tk.LEFT)
        self.gen_name = tk.Entry(f5, font=('Arial', 11), width=15)
        self.gen_name.insert(0, "custom_instance")
        self.gen_name.pack(side=tk.LEFT, padx=5)
        
        # Bouton générer
        btn_gen = tk.Button(gen_window, text="Generer", 
                           command=lambda: self.create_instance(gen_window),
                           font=('Arial', 12), bg='#27ae60', fg='white',
                           padx=30, pady=8, cursor='hand2', relief=tk.FLAT)
        btn_gen.pack(pady=20)
        
        # Status
        self.gen_status = tk.Label(gen_window, text="", font=('Arial', 10), bg='white', fg='#666')
        self.gen_status.pack()
        
    def create_instance(self, window):
        """Cree une instance avec les parametres donnes"""
        import random
        
        try:
            n = int(self.gen_n.get())
            capacity = int(self.gen_w.get())
            w_max = int(self.gen_wmax.get())
            v_max = int(self.gen_vmax.get())
            name = self.gen_name.get().strip() or "custom_instance"
            
            if n <= 0 or capacity <= 0 or w_max <= 0 or v_max <= 0:
                raise ValueError("Toutes les valeurs doivent etre positives")
            
            # Créer le dossier data/generated si nécessaire
            gen_dir = os.path.join(os.path.dirname(__file__), 'data', 'generated')
            os.makedirs(gen_dir, exist_ok=True)
            
            # Générer les items (poids, valeur)
            weights = []
            values = []
            for _ in range(n):
                weight = random.randint(1, w_max)
                value = random.randint(1, v_max)
                weights.append(weight)
                values.append(value)
            
            # Calculer la solution optimale avec DP
            optimal_value = self._compute_optimal_dp(weights, values, capacity)
            
            # Sauvegarder avec l'optimal en ligne 3
            filepath = os.path.join(gen_dir, f"{name}.txt")
            with open(filepath, 'w') as f:
                f.write(f"{n}\n")
                f.write(f"{capacity}\n")
                f.write(f"{optimal_value}\n")  # Solution optimale
                for w, v in zip(weights, values):
                    f.write(f"{w} {v}\n")
            
            self.gen_status.config(text=f"Instance creee: {name}.txt (OPT={optimal_value})", fg='#27ae60')
            self.status.config(text=f"Nouvelle instance: data/generated/{name}.txt")
            
        except ValueError as e:
            self.gen_status.config(text=f"Erreur: {str(e)}", fg='#e74c3c')
        except Exception as e:
            self.gen_status.config(text=f"Erreur: {str(e)}", fg='#e74c3c')
    
    def _compute_optimal_dp(self, weights, values, capacity):
        """Calcule la solution optimale avec programmation dynamique"""
        n = len(weights)
        # Utiliser un tableau 1D pour économiser la mémoire
        dp = [0] * (capacity + 1)
        
        for i in range(n):
            w, v = weights[i], values[i]
            # Parcourir à l'envers pour éviter de réutiliser le même item
            for c in range(capacity, w - 1, -1):
                dp[c] = max(dp[c], dp[c - w] + v)
        
        return dp[capacity]


def main():
    root = tk.Tk()
    app = KnapsackGUI(root)
    
    # Centrer
    root.update_idletasks()
    x = (root.winfo_screenwidth() - 1100) // 2
    y = (root.winfo_screenheight() - 700) // 2
    root.geometry(f"1100x700+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
