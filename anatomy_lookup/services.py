import os
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch

class AnatomyLookup:
    embs_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../resources/onto_embs.pt')
    def __init__(self):
        self.model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

        # checking if the term embeddings available
        if not os.path.exists(AnatomyLookup.embs_file):
            # download term embeddings
            logging.warning("Term embeddings are not available.")
            AnatomyLookup.update_terms()

        # loading term embeddings
        self.onto_ids, self.onto_terms, self.onto_labels, self.onto_embs = torch.load(AnatomyLookup.embs_file)

    def build_indexes(self, file_path:str):
        """
        Building UBERON and ILX term embedding
        file_path = a ttl file path or a directory containing ttl files
        The files can be obtained from https://github.com/SciCrunch/NIF-Ontology/releases
        """
        from rdflib import Graph
        # getting all ttl files
        filenames = []
        if os.path.isfile(file_path):
            filenames += [file_path] if file_path.endswith('ttl') else []
        elif os.path.isdir(file_path):
            for filename in next(os.walk(file_path), (None, None, []))[2]:
                if filename.endswith('ttl'):
                    filenames += [os.path.join(file_path, filename)]
        if len(filenames) == 0:
            logging.error('No ttl file available')
            return

        # parsing all ttl files
        g = Graph()
        for filename in tqdm(filenames):
            g.parse(filename)

        # getting UBERON and ILX terms
        onto_ids = []
        onto_terms = []
        onto_labels = {}
        for s, p, o in tqdm(g):
            if ('UBERON' in s or 'ilx_' in s) and ('label' in p.casefold() or 'synonym' in p.casefold()):
                onto_ids += [str(s)]
                onto_terms += [str(o)]
                if 'label' in p.casefold():
                    onto_labels[str(s)] = str(o)

        # generating term embeddings
        onto_embs = self.model.encode(onto_terms, show_progress_bar=True, convert_to_tensor=True)

        # saving embeddings into a file
        data_embds = (onto_ids, onto_terms, onto_labels, onto_embs)
        torch.save(data_embds, AnatomyLookup.embs_file)

    def update_terms():
        import requests
        logging.warning("... downloading from server")
        r = requests.get('https://auckland.figshare.com/ndownloader/files/38944175')
        with open(AnatomyLookup.embs_file, 'wb') as f:
            f.write(r.content)

    def search(self, query:str):
        query_emb = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_emb, self.onto_embs)[0]
        top_results = torch.topk(cos_scores, k=1)
        score = top_results[0][0].item()
        ids = top_results[1][0].item()
        url = self.onto_ids[ids]
        return (url, self.onto_labels[url], score)
        