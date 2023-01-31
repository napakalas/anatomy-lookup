import os
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch

class AnatomyLookup:
    embs_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources')
    embs_file = os.path.join(embs_folder, 'onto_embs.pt')
    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        # checking if the term embeddings available
        if not os.path.exists(AnatomyLookup.embs_file):
            # creating folder if not available
            if not os.path.exists(AnatomyLookup.embs_folder):
                os.mkdir(AnatomyLookup.embs_folder)
            # download term embeddings
            logging.warning("Term embeddings are not available.")
            self.update_terms()

        # loading term embeddings
        self.__load_embedding_file()

    def __load_embedding_file(self):
        self.onto_ids, self.onto_labels, self.onto_embs = torch.load(AnatomyLookup.embs_file)

    def build_indexes(self, file_path:str):
        """
        Building UBERON and ILX term embedding
        file_path = a ttl file path or a directory containing ttl files
        The files can be obtained from https://github.com/SciCrunch/NIF-Ontology/releases
        """
        import rdflib
        import re
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
        g = rdflib.Graph()
        for filename in tqdm(filenames):
            g.parse(filename)

        # getting UBERON and ILX terms with label predicate
        onto_ids = []
        onto_terms = []
        onto_pos = {}
        onto_labels = {}
        p_label = 'http://www.w3.org/2000/01/rdf-schema#label'
        for s, o in tqdm(g.subject_objects(rdflib.util.URIRef(p_label))):
            if ('UBERON' in s or 'ilx_' in s):
                onto_labels[str(s)] = str(o)
                onto_pos[str(s)] = len(onto_ids)
                onto_ids += [str(s)]
                onto_terms +=  [str(o)]

        # generating label term embeddings
        onto_embs = self.model.encode(onto_terms, show_progress_bar=True, convert_to_tensor=True)

        # getting terms from synonym type predicate
        p_synonyms = ['http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym',
              'http://www.geneontology.org/formats/oboInOwl#hasExactSynonym',
              'http://uri.neuinfo.org/nif/nifstd/readable/synonym',
              'http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym',
              'http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym',
             ]
        onto_terms_syn = []
        onto_ids_syn = []
        for p in p_synonyms:
            for s, o in tqdm(g.subject_objects(rdflib.util.URIRef(p))):
                if ('UBERON' in s or 'ilx_' in s):
                    pred = re.sub('([A-Z])', r' \1', str(p).split('#')[-1].split('/')[-1]).lower()
                    onto_ids_syn += [str(s)]
                    onto_terms_syn += [str(o)]

        # generating synonym term embeddings
        onto_embs_syn = self.model.encode(onto_terms_syn, show_progress_bar=True, convert_to_tensor=True)

        # modify synonym term embeddings by adding with label term embeddings
        for i, ids in enumerate(onto_ids_syn):
            pos = onto_pos[ onto_ids_syn[i]]
            onto_embs_syn[i]  += onto_embs[pos]

        # combining synonym term embeddings to all embeddings
        onto_embs = torch.cat([onto_embs,onto_embs_syn])
        onto_ids += onto_ids_syn   

        # saving embeddings into a file
        data_embds = (onto_ids, onto_labels, onto_embs)
        torch.save(data_embds, AnatomyLookup.embs_file)

        # loading term embeddings
        self.__load_embedding_file()

    def update_terms(self):
        import requests
        import json

        # get newest download link
        article_id = '21952595'
        url = 'https://api.figshare.com/v2/articles/{}/files'.format(article_id)
        headers = {'Content-Type': 'application/json'}
        response = requests.request('GET', url, headers=headers)
        file_url = json.loads(response.text)[0]['download_url']

        # downloading the file
        logging.warning("... downloading from server")
        r = requests.get(file_url)
        with open(AnatomyLookup.embs_file, 'wb') as f:
            f.write(r.content)
        
        # loading term embeddings
        self.__load_embedding_file()

    def search(self, query:str):
        query_emb = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_emb, self.onto_embs)[0]
        top_results = torch.topk(cos_scores, k=1)
        score = top_results[0][0].item()
        ids = top_results[1][0].item()
        url = self.onto_ids[ids]
        return (url, self.onto_labels[url], score)
        
    def close(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        del self.model
        return None