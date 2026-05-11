import torch
from torch_geometric.data import Data, Batch
from typing import List, Union, Optional
import networkx as nx
from torch_geometric.utils import from_networkx

class GraphDataLoader:
    def __init__(self, batch_size: int = 32):
        """
        Initialisiert den Dataloader für Graphen.
        
        Args:
            batch_size: Die Anzahl an Graphen, die gebündelt (gebatcht) werden sollen.
        """
        self.batch_size = batch_size
        self.buffer: List[Data] = []
        
    def add_graph(self, graph: Union[Data, nx.DiGraph]):
        """
        Nimmt einen Graphen an (entweder als PyTorch Geometric Data oder NetworkX DiGraph).
        Falls es ein NetworkX Graph ist, wird er konvertiert.
        Anschließend wird der Graph im Buffer gespeichert.
        """
        if isinstance(graph, nx.DiGraph):
            # Konvertiert den Graphen in ein PyG Data-Objekt.
            # Hinweis: Wenn im Preprocessor Features an Knoten angehängt wurden,
            # übernimmt from_networkx diese in der Regel als separate Tensoren (z.B. data.currentX).
            data = from_networkx(graph)
        elif isinstance(graph, Data):
            data = graph
        else:
            raise TypeError("Graph muss vom Typ torch_geometric.data.Data oder networkx.DiGraph sein.")
            
        self.buffer.append(data)
        
    def has_batch(self) -> bool:
        """
        Prüft, ob der Buffer genügend Graphen für einen vollständigen Batch enthält.
        """
        return len(self.buffer) >= self.batch_size
        
    def get_batch(self) -> Optional[Batch]:
        """
        Gibt einen Batch von Graphen zurück und entfernt diese aus dem Buffer.
        Gibt None zurück, falls nicht genügend Graphen vorhanden sind.
        """
        if not self.has_batch():
            return None
            
        # Nimm die ersten 'batch_size' Graphen
        batch_list = self.buffer[:self.batch_size]
        self.buffer = self.buffer[self.batch_size:]
        
        # Erstelle den Batch
        return Batch.from_data_list(batch_list)
        
    def flush(self) -> Optional[Batch]:
        """
        Erzeugt einen Batch aus allen verbleibenden Graphen im Buffer.
        Wird typischerweise am Ende einer Epoche oder eines Durchlaufs aufgerufen,
        wenn der Rest nicht die volle batch_size erreicht.
        """
        if len(self.buffer) == 0:
            return None
            
        batch_list = self.buffer
        self.buffer = []
        
        return Batch.from_data_list(batch_list)
        
    def __len__(self):
        """Anzahl der aktuell im Buffer befindlichen Graphen."""
        return len(self.buffer)
