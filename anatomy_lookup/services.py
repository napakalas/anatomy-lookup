import os
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import requests
import rdflib
from rdflib.namespace import OWL
import json
import xlsxwriter

from typing import Optional

#===============================================================================

RESOURCE_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources')
EMBEDDING_FILE = 'onto_embs.pt'
SCKAN_RELEASE = 'https://api.github.com/repos/SciCrunch/NIF-Ontology/releases'
SCKAN_ASSET = 'release'
SCKAN_PICKLE = 'rdflib.graph'
SPELLING_FILE = 'spelling_embs.pt'
ONTO_HIERARCHY = 'onto_hierarchy.json'
DATA_LOG = 'data_log.json'
BERTModel = 'gsarti/biobert-nli'
# BERTModel = 'multi-qa-MiniLM-L6-cos-v1'

#===============================================================================
namespaces = { 
    'UBERON': 'http://purl.obolibrary.org/obo/UBERON_',
    'ILX': 'http://uri.interlex.org/base/ilx_',
    # 'BFO': 'http://purl.obolibrary.org/obo/BFO_',
    # 'CHEBI': 'http://purl.obolibrary.org/obo/CHEBI_',
    # 'CL': 'http://purl.obolibrary.org/obo/CL_',
    # 'GO': 'http://purl.obolibrary.org/obo/GO_',
    # 'NCBITaxon': 'http://purl.obolibrary.org/obo/NCBITaxon_',
    # 'NBO': 'http://purl.obolibrary.org/obo/NBO_',
    # 'PATO': 'http://purl.obolibrary.org/obo/PATO_',
    # 'PR': 'http://purl.obolibrary.org/obo/PR_',
    # 'CARO': 'http://purl.obolibrary.org/obo/CARO_',
    # 'FMA': 'http://purl.org/sig/ont/fma/fma'
}

def get_uriref(uri_or_curie:str) -> rdflib.URIRef:
    if 'http' in uri_or_curie:
        return rdflib.URIRef(uri_or_curie)
    elif ':' in uri_or_curie:
        parts = uri_or_curie.split(':', 1)
        if parts[0] in namespaces:
            return rdflib.URIRef(namespaces[parts[0]] + parts[1])

def get_uri(curie_or_uriref) -> Optional[str]:
    if str(curie_or_uriref).startswith('http'):
        return str(curie_or_uriref)
    elif ':' in curie_or_uriref:
        parts = curie_or_uriref.split(':', 1)
        if parts[0] in namespaces:
            return namespaces[parts[0]] + parts[1]

def get_curie(uri_or_uriref) -> Optional[str]:
    uri = str(uri_or_uriref)
    for k, v in namespaces.items():
        if uri.startswith(v):
            return k + ':' + uri.split(v)[-1]

#===============================================================================

def __download_latest_SCKAN(sckan_release=None):
    import tarfile
    import zipfile
    import shutil

    releases = requests.get(SCKAN_RELEASE, timeout=10)
    selected_release = None

    # select SCKAN release based on sckan_release or the most recent release
    if sckan_release:
        for release in releases.json():
            if release['tag_name'] == sckan_release:
                selected_release = release

    if not sckan_release: # getting the latest release from SCKAN repository
        selected_release = max(releases.json(), key=lambda x:x['published_at'])
        sckan_release = selected_release['tag_name']

    # stop when selectedrelease is not identified
    if selected_release is None:
        import sys
        sys.exit()

    # log selected release
    logging.info(f"The extracted SCKAN version is {sckan_release}")
    
    # selecting asset to be dowloaded
    file_url = next((asset['browser_download_url'] for asset in selected_release['assets'] 
                        if SCKAN_ASSET in asset['browser_download_url'].split('/')[-1]), None)
    file_url = file_url if file_url != None else selected_release['zipball_url']
    file_path = os.path.join(RESOURCE_FOLDER, file_url.split('/')[-1])
    logging.info('Downloading SCKAN file from:', file_url)
    r = requests.get(file_url)
    with open(file_path, 'wb') as f:
        f.write(r.content)
    
    # deleting old files
    extracted_path = os.path.join(RESOURCE_FOLDER, SCKAN_ASSET)
    for file in os.listdir(extracted_path):
        file_or_folder = os.path.join(extracted_path, file)
        if os.path.isfile(file_or_folder) or os.path.islink(file_or_folder):
            os.unlink(file_or_folder)
        elif os.path.isdir(file_or_folder):
            shutil.rmtree(file_or_folder)

    # extracting zip or tar.gz file
    if file_path.endswith('tar.gz'):
        arc_file = tarfile.open(file_path, 'r:gz')
    elif file_path.endswith('zip'):
        arc_file = zipfile.ZipFile(file_path, 'r')
    arc_file.extractall(extracted_path)
    arc_file.close()
    
    # deleting zip file and non ttl files
    os.remove(file_path)
    for file in __get_ttl(extracted_path)[1]:
        os.remove(file)
    return extracted_path


def __get_ttl(path):
    """returning list of ttl_files and non ttl_files
    """
    ttl_files = []
    other_files = []
    for folder in next(os.walk(path), (None, None, []))[1:]:
        for p in folder:
            np = os.path.join(path, p)
            if os.path.isdir(np):
                tmp =__get_ttl(np)
                ttl_files += tmp[0]
                other_files += tmp[1]
            elif p.endswith('ttl'):
                ttl_files += [np]
            else:
                other_files += [np]
    return [ttl_files, other_files]

def get_SCKAN_graph(sckan_release: Optional[str]=None):
    # downloading SCKAN if file_path=None
    file_path = __download_latest_SCKAN(sckan_release)
    
    # getting all ttl files
    filenames = __get_ttl(file_path)[0]
    if len(filenames) == 0:
        logging.warning('No ttl file available.')
        return

    # parsing all ttl files
    g = rdflib.Graph()
    for filename in tqdm(filenames):
        try:
            g.parse(filename)
        except Exception as e:
            logging.error(f'{e} - Cannot load file: {filename}')

    return g

#===============================================================================

class AnatomyLookup:
    embs_file = os.path.join(RESOURCE_FOLDER, EMBEDDING_FILE)
    spell_file = os.path.join(RESOURCE_FOLDER, SPELLING_FILE)
    hierarchy_file = os.path.join(RESOURCE_FOLDER, ONTO_HIERARCHY)
    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.__model = SentenceTransformer(BERTModel, device='cpu')
        # checking if the term embeddings available
        # creating folder if not available
        if not os.path.exists(RESOURCE_FOLDER):
            os.mkdir(RESOURCE_FOLDER)
        self.__check_current_term_files()

        # loading term embeddingsz
        self.__load_embedding_file()

    def __load_embedding_file(self):
        device = torch.device('cpu')
        self.__onto_ids, self.__onto_labels, self.__onto_embs = torch.load(AnatomyLookup.embs_file, device)
        self.__spell_phrases, self.__spell_embs = torch.load(AnatomyLookup.spell_file, device)
        with open(AnatomyLookup.hierarchy_file, 'r') as fp:
            self.__onto_hierarchy = json.load(fp)

    def build_indexes(self, sckan_release:Optional[str]=None):
        """
        Building UBERON and ILX term embedding -> (now try to cover all classes, not just UBERON and ILX)
        sckan_release = SCKAN release to be dowloaded from repository
        The files can be obtained from https://github.com/SciCrunch/NIF-Ontology/releases
        """

        g = get_SCKAN_graph(sckan_release)

        if g is None:
            return

        # getting ontology terms with label predicate
        predicates = ['http://www.w3.org/2000/01/rdf-schema#label']
        onto_terms, onto_ids, onto_labels = self.__get_terms(g, predicates)

        # getting ontology terms labelr as annotations, usually for ILX
        sparql = """
            SELECT DISTINCT ?s ?o {
                ?s <https://apinatomy.org/uris/readable/annotates>[
                    rdfs:label ?o;
                    rdf:type <https://apinatomy.org/uris/elements/Material>
                ]
        }
        """
        qres = g.query(sparql)
        for row in qres:
            if any([ns in str(row.s) for ns in namespaces.values()]):
                if str(row.s) not in onto_labels:
                    onto_terms += [str(row.o)]
                    onto_ids += [str(row.s)]
                    onto_labels[str(row.s)] = [str(row.o)]

        # getting terms with exact synonym predicate
        predicates = ['http://www.geneontology.org/formats/oboInOwl#hasExactSynonym']
        onto_terms_syn, onto_ids_syn, _ = self.__get_terms(g, predicates)
        onto_terms += onto_terms_syn
        onto_ids += onto_ids_syn

        # generating label term embeddings
        onto_embs = self.__model.encode(onto_terms, show_progress_bar=True, convert_to_tensor=True)

        # getting terms with nifstd synonym predicate (sometime ambiguous)
        predicates = ['http://uri.neuinfo.org/nif/nifstd/readable/synonym',
                ]
        onto_terms_syn, onto_ids_syn, duplicates = self.__get_terms(g, predicates, duplicate_check=True)
        onto_embs_syn = []
        label_weight = 0.5
        for term, idx in tqdm(list(zip(onto_terms_syn, onto_ids_syn))):
            if term in onto_terms or term in duplicates:
                onto_embs_syn += self.__model.encode([term], show_progress_bar=False, convert_to_tensor=True) + onto_embs[onto_ids.index(idx)] * label_weight
            else:
                onto_embs_syn += self.__model.encode([term], show_progress_bar=False, convert_to_tensor=True)
            onto_terms += [term]
            onto_ids += [idx]
            
        # getting terms from other synonym type predicate
        predicates = ['http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym',
                'http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym',
                'http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym',
                ]
        onto_terms_syn, onto_ids_syn, duplicates = self.__get_terms(g, predicates, duplicate_check=True)
        synonym_weight = 0.5
        for term, idx in tqdm(list(zip(onto_terms_syn, onto_ids_syn))):
            onto_embs_syn += self.__model.encode([term], show_progress_bar=False, convert_to_tensor=True) * synonym_weight + onto_embs[onto_ids.index(idx)]
            onto_terms += [term]
            onto_ids += [idx]
        
        onto_embs = torch.cat([onto_embs,torch.stack(onto_embs_syn)])
        # saving embeddings into a file
        data_embds = (onto_ids, onto_labels, onto_embs)
        torch.save(data_embds, AnatomyLookup.embs_file)

        # creating embeddings for spell correction
        onto_terms = list(set(onto_terms))
        phrase_embs = self.__model.encode(onto_terms, show_progress_bar=True, convert_to_tensor=True)
        data_phrase_embs = (onto_terms, phrase_embs)
        torch.save(data_phrase_embs, AnatomyLookup.spell_file)

        # extracting ontology's hierarchical structure
        self.__extract_ontology_hierarchy(g)

        # loading term embeddings
        self.__load_embedding_file()

    def __get_terms(self, g:rdflib.Graph, predicates:list, duplicate_check=False, include_deprecated=False, predicate_con='|'):
        terms = []
        ids = []
        labels = {}
        path = rdflib.URIRef(predicates[0])
        if len(predicates) > 1:
            for p in predicates[1:]:
                if predicate_con == '|':
                    path = path | rdflib.URIRef(p)
                elif predicate_con == '/':
                    path = path / rdflib.URIRef(p)
        for s, o in g.subject_objects(path, unique=True):
            if any([str(s).startswith(ns) for ns in namespaces.values()]):
                if include_deprecated:
                    ids += [str(s)]
                    terms += [str(o)]
                    labels[str(s)] = str(o)
                elif len(list(g.objects(s, OWL.deprecated))) == 0:
                    ids += [str(s)]
                    terms += [str(o)]
                    labels[str(s)] = str(o)
        if duplicate_check:
            labels = set([t for t in terms if terms.count(t) > 1])
        return terms, ids, labels

    def __extract_ontology_hierarchy(self, g:rdflib.Graph):
        path = rdflib.RDFS.subClassOf|(rdflib.RDFS.subClassOf/rdflib.OWL.someValuesFrom)
        ids = {}
        for s, o in tqdm(g.subject_objects(path, unique=True)):
            s = str(s)
            o = str(o)
            if any(s.startswith(ns) and o.startswith(ns) for ns in namespaces.values()):
                s = str(s)
                o = str(o)
                if s not in ids:
                    ids[s] = {'parents': [], 'children':[]}
                if o not in ids:
                    ids[o] = {'parents': [], 'children':[]}
                if s!= o:
                    ids[s]['parents'] += [] if o in ids[s]['parents'] else [o]
                    ids[o]['children'] += [] if s in ids[o]['children'] else [s]
        with open(AnatomyLookup.hierarchy_file, 'w') as fp:
            json.dump(ids, fp)

    def __check_current_term_files(self):

        if os.path.exists(file_log:=os.path.join(RESOURCE_FOLDER, DATA_LOG)):
            with open(file_log, 'r') as fp:
                data_log = json.load(fp)
        else:
            data_log = {"version": 0, "published_at":""}
        # check current data
        if data_log['version'] == 0:
            # download term embeddings
            logging.warning(" Term embeddings are not available.")
            self.update_terms()
            return
        article_id = '21952595'
        url = 'https://api.figshare.com/v2/articles/{}'.format(article_id)
        headers = {'Content-Type': 'application/json'}
        response = requests.request('GET', url, headers=headers)
        latest_version = response.json()['version']
        if data_log['version'] < latest_version:
            logging.warning('Local indexes are obsolete. Please run update_terms() function')

    def update_terms(self):
        # get newest download link
        
        article_id = '21952595'
        url = 'https://api.figshare.com/v2/articles/{}'.format(article_id)
        headers = {'Content-Type': 'application/json'}
        response = requests.request('GET', url, headers=headers)
        response = response.json()
        
        for file_url in response['files']:
            # downloading the file
            logging.warning(" ... downloading from server: " + file_url['name'])
            r = requests.get(file_url['download_url'], timeout=10)
            save_to = os.path.join(RESOURCE_FOLDER, file_url['name'])
            with open(save_to, 'wb') as f:
                f.write(r.content)
        
        # save to log
        data_log = {
            'version': response['version'],
            'published_at': response['timeline']['posted']
        }
        with open(os.path.join(RESOURCE_FOLDER, DATA_LOG), 'w') as fp:
            json.dump(data_log, fp)

        # loading term embeddings
        self.__load_embedding_file()

    def __get_query_emb(self, query:str, force=False, refine=True):
        unique_abbr={' g.': ' ganglion'}
        if force:
            for k, v in unique_abbr.items():
                query = query.replace(k, v)
        query_emb = self.__model.encode(query, show_progress_bar=False,  convert_to_tensor=True)
        if not refine:
            return query_emb
        # refine query using available phrases. in a case of misspelling
        cos_scores = util.cos_sim(query_emb, self.__spell_embs)[0]
        top_results = torch.topk(cos_scores, k=1)
        score = top_results[0][0].item()
        if score >= 0.88 and score <1:
            query = self.__spell_phrases[top_results[1][0].item()]
            query_emb = self.__model.encode(query, show_progress_bar=False, convert_to_tensor=True)
        # return emb
        return query_emb
        
    def search_candidates(self, query:str, k:int, uri_candidates:Optional[list]=None, force=False, refine=True) -> list[tuple[str, str, float]]:
        """
        k -> the number of results returned, between 1 and 10
        """
        query_emb = self.__get_query_emb(query, force=force, refine=refine)
        
        if uri_candidates == None:
            uri_candidates = self.__onto_ids
        
        uri_candidates = [get_uri(c) for c in uri_candidates]

        cos_scores = util.cos_sim(query_emb, self.__onto_embs)[0]
        
        k = k if len(uri_candidates) > k else len(uri_candidates)       # type: ignore
        k = 10 if k>10 else k
        top_results = torch.topk(cos_scores, k=100)
        
        results = []
        record = set()
        for score, idx in zip(top_results[0], top_results[1]):
            url = self.__onto_ids[idx.item()]
            if url not in record:
                if url in uri_candidates:
                    results += [(get_curie(url), self.__onto_labels[url], score.item())]
                    record.update([url])
            if len(record) == k:
                break
        return results

    def __get_parent_or_children(self, idx, is_parent:bool, expand:bool):
        idx = get_uri(idx)
        if is_parent:
            get_type = 'parents'
        elif is_parent==False:
            get_type = 'children'
            
        lines = set()
        if idx in self.__onto_hierarchy:
            attaches = self.__onto_hierarchy[idx][get_type]
            while True:
                lines.update(attaches)
                tmp_lines = set()
                for attach in attaches:
                    tmp_lines.update(self.__onto_hierarchy[attach][get_type])
                tmp_lines = tmp_lines-lines
                if len(tmp_lines)==0:
                    return lines
                attaches = tmp_lines
        return lines

    def get_descendant(self, uri):
        return self.__get_parent_or_children(uri, is_parent=False)

    def get_ancestor(self, uri):
        return self.__get_parent_or_children(uri, is_parent=True)

    def is_uri_connected(self, uri1, uri2) -> bool:
        """
        checking whether a URI is a subclass of another URI
        """
        if uri1 in self.get_ancestor(uri2):
            return True
        if uri1 in self.get_descendant(uri2):
            return True
        return False

    def search(self, query:str, force=False, refine=True):
        # searching for ontology term
        results = self.search_candidates(query, k=1, force=force, refine=refine)
        return results[0]
    
    def search_with_scope(self, query:str, scope: str|list[str], k:int=5, threshold=0.8, force=False, refine=True):
        if isinstance(scope, str):
            idx_scope, _, score_scope = self.search(scope)
            if score_scope < threshold:
                logging.info("Scope is not available, the score lower than 0.8")
                return []
            return self.search_candidates(query, k, list(self.get_descendant(idx_scope)), force=force, refine=refine)
        elif isinstance(scope, list):
            descendants = set()
            for sc in scope:
                idx_scope, _, score_scope = self.search(sc)
                if score_scope >= threshold:
                    descendants.update(list(self.get_descendant(idx_scope)))
            if len(descendants) == 0:
                logging.info("Scope is not available, the score lower than 0.8")
                return []
            return self.search_candidates(query, k, list(descendants), force=force)

    def close(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        del self.__model
        return None

    
#===============================================================================

class AnatomyAnnotator:

    def __init__(self):
        self.__lookup = AnatomyLookup()
        self.__data = {}

    def annotate(self, data_file:str, search_attr:str, scope_attrs:list, threshold=0.8, force=False, refine=True):
        """
        A method to annotate data to ontology terms
        data_file -> usually a json file containing systems, nerves, organs, and ftus
        search_attr -> the attribute to be annotated
        scope_attrs -> list of attribute to limit annotation
        threshold -> the minimum score to make sure the annotation is correct
        """
        with open(data_file, 'r') as fp:
            data = json.load(fp)

        for group in data.values():
            for item in tqdm(group):
                properties = self.__lookup.search(item[search_attr], force=force, refine=refine)
                if properties[2] >= threshold:
                    item['properties'] = {'models': get_curie(properties[0]),
                                          'label': properties[1],
                                          'confidence': properties[2]}
                    item['term'] = get_curie(properties[0])
                elif len(set(scope_attrs)&set(item.keys()))==1:
                    key = list(set(scope_attrs)&set(item.keys()))[0]
                    l_props = self.__lookup.search_with_scope(item[search_attr],item[key], force=force, refine=refine)
                    if len(l_props) > 0:
                        item['suggestions'] = []
                        for properties in l_props:
                             item['suggestions'] += [{'models': get_curie(properties[0]),
                                                      'label': properties[1],
                                                      'confidence': properties[2]}]
        self.__data = data

    def save_to_xlsx(self, file_name):
        workbook = xlsxwriter.Workbook(file_name)
        workbook.set_size(1600, 1200)
        header_format = workbook.add_format({'bold': True,
                                             'align': 'left',
                                             'valign': 'top',
                                             'fg_color': '#80C080',
                                             'border': 1,
                                             })
        hidden = workbook.add_format({
            'hidden': True,
            'bg_color': '#E0E0E0',
            'left': 1,
            'border': 1,
            'border_color': '#C0C0C0',
            })
        locked = workbook.add_format({
            'locked': True,
            'border': 1,
            'bg_color': '#E0E0E0',
            'border_color': '#C0C0C0',
            })
        locked_name = workbook.add_format({
            'locked': True,
            'border': 1,
            'bg_color': '#EBF1DE',
            'border_color': '#C0C0C0',
            })
        unlocked = workbook.add_format({'locked': False})

        if 'systems' in self.__data:
            worksheet = workbook.add_worksheet('OrganSystems')
            worksheet.protect()
            worksheet.freeze_panes(1, 0)
            worksheet.set_row(0, 20, header_format)
            worksheet.set_column('A:A', 32, locked_name)
            worksheet.set_column('B:D', 24, unlocked)
            worksheet.set_column('E:S', 24, locked)
            worksheet.write_string(0, 0, 'Organ System Name')
            worksheet.write_string(0, 1, 'Model')
            worksheet.write_string(0, 2, 'Label')
            worksheet.write_string(0, 3, 'Confidence')
            for i in range(1,4):
                worksheet.write_string(0, i*3+1, 'Suggestion Model {}'.format(i))
                worksheet.write_string(0, i*3+2, 'Suggestion Label {}'.format(i))
                worksheet.write_string(0, i*3+3, 'Suggestion Confidence {}'.format(i))
            for row, item in enumerate(self.__data['systems']):
                worksheet.write_string(row + 1, 0, item['name'])
                if 'properties' in item:
                    worksheet.write_string(row + 1, 1, item['properties']['models'])
                    worksheet.write_string(row + 1, 2, item['properties']['label'])
                    worksheet.write_string(row + 1, 3, str(item['properties']['confidence']))
                elif 'suggestions' in item:
                    for col, suggestion in enumerate(item['suggestions']):
                        worksheet.write_string(row + 1, col*3+4, suggestion['models'])
                        worksheet.write_string(row + 1, col*3+5, suggestion['label'])
                        worksheet.write_string(row + 1, col*3+6, str(suggestion['confidence']))

        if 'nerves' in self.__data:
            worksheet = workbook.add_worksheet('OrganNerves')
            worksheet.protect()
            worksheet.freeze_panes(1, 0)
            worksheet.set_row(0, 20, header_format)
            worksheet.set_column('A:A', 32, locked_name)
            worksheet.set_column('B:D', 24, unlocked)
            worksheet.set_column('E:S', 24, locked)
            worksheet.write_string(0, 0, 'Nerve System Name')
            worksheet.write_string(0, 1, 'Model')
            worksheet.write_string(0, 2, 'Label')
            worksheet.write_string(0, 3, 'Confidence')
            for i in range(1,4):
                worksheet.write_string(0, i*3+1, 'Suggestion Model {}'.format(i))
                worksheet.write_string(0, i*3+2, 'Suggestion Label {}'.format(i))
                worksheet.write_string(0, i*3+3, 'Suggestion Confidence {}'.format(i))
            for row, item in enumerate(self.__data['nerves']):
                worksheet.write_string(row + 1, 0, item['name'])
                if 'properties' in item:
                    worksheet.write_string(row + 1, 1, item['properties']['models'])
                    worksheet.write_string(row + 1, 2, item['properties']['label'])
                    worksheet.write_string(row + 1, 3, str(item['properties']['confidence']))
                elif 'suggestions' in item:
                    for col, suggestion in enumerate(item['suggestions']):
                        worksheet.write_string(row + 1, col*3+4, suggestion['models'])
                        worksheet.write_string(row + 1, col*3+5, suggestion['label'])
                        worksheet.write_string(row + 1, col*3+6, str(suggestion['confidence']))

        if 'organs' in self.__data:
            worksheet = workbook.add_worksheet('Organs')
            worksheet.protect()
            worksheet.freeze_panes(1, 0)
            worksheet.set_row(0, 20, header_format)
            worksheet.set_column('A:B', 32, locked_name)
            worksheet.set_column('C:E', 24, unlocked)
            worksheet.set_column('F:T', 24, locked)
            worksheet.write_string(0, 0, 'Organ Name')
            worksheet.write_string(0, 1, 'System')
            worksheet.write_string(0, 2, 'Model')
            worksheet.write_string(0, 3, 'Label')
            worksheet.write_string(0, 4, 'Confidence')
            for i in range(1,4):
                worksheet.write_string(0, i*3+2, 'Suggestion Model {}'.format(i))
                worksheet.write_string(0, i*3+3, 'Suggestion Label {}'.format(i))
                worksheet.write_string(0, i*3+4, 'Suggestion Confidence {}'.format(i))
            for row, item in enumerate(self.__data['organs']):
                worksheet.write_string(row + 1, 0, item['name'])
                worksheet.write_string(row + 1, 1, ', '.join(item['systems']))
                if 'properties' in item:
                    worksheet.write_string(row + 1, 2, item['properties']['models'])
                    worksheet.write_string(row + 1, 3, item['properties']['label'])
                    worksheet.write_string(row + 1, 4, str(item['properties']['confidence']))
                elif 'suggestions' in item:
                    for col, suggestion in enumerate(item['suggestions']):
                        worksheet.write_string(row + 1, col*3+5, suggestion['models'])
                        worksheet.write_string(row + 1, col*3+6, suggestion['label'])
                        worksheet.write_string(row + 1, col*3+7, str(suggestion['confidence']))
        
        if 'ftus' in self.__data:
            worksheet = workbook.add_worksheet('FTUs')
            worksheet.protect()
            worksheet.freeze_panes(1, 0)
            worksheet.set_row(0, 20, header_format)
            worksheet.set_column('A:B', 32, locked_name)
            worksheet.set_column('C:E', 24, unlocked)
            worksheet.set_column('F:T', 24, locked)
            worksheet.write_string(0, 0, 'FTU Name')
            worksheet.write_string(0, 1, 'Organ')
            worksheet.write_string(0, 2, 'Model')
            worksheet.write_string(0, 3, 'Label')
            worksheet.write_string(0, 4, 'Confidence')
            for i in range(1,4):
                worksheet.write_string(0, i*3+2, 'Suggestion Model {}'.format(i))
                worksheet.write_string(0, i*3+3, 'Suggestion Label {}'.format(i))
                worksheet.write_string(0, i*3+4, 'Suggestion Confidence {}'.format(i))
            for row, item in enumerate(self.__data['ftus']):
                worksheet.write_string(row + 1, 0, item['name'])
                worksheet.write_string(row + 1, 1, item['organ'])
                if 'properties' in item:
                    worksheet.write_string(row + 1, 2, item['properties']['models'])
                    worksheet.write_string(row + 1, 3, item['properties']['label'])
                    worksheet.write_string(row + 1, 4, str(item['properties']['confidence']))
                elif 'suggestions' in item:
                    for col, suggestion in enumerate(item['suggestions']):
                        worksheet.write_string(row + 1, col*3+5, suggestion['models'])
                        worksheet.write_string(row + 1, col*3+6, suggestion['label'])
                        worksheet.write_string(row + 1, col*3+7, str(suggestion['confidence']))

        workbook.close()


    def save_to_json(self, file_name):
        with open(file_name, 'w') as fp:
            json.dump(self.__data, fp)

    def get_results(self):
        return self.__data

#===============================================================================

class AnatomyValidator:
    
    def __init__(self):
        self.__graph = get_SCKAN_graph(os.path.join(RESOURCE_FOLDER, SCKAN_ASSET))
        self.__lookup = AnatomyLookup()
        
    def is_uri_linked(self, uri1, uri2) -> bool:
        """
        checking whether a URI is a subclass of another URI
        """
        uriref1 = get_uriref(uri1)
        uriref2 = get_uriref(uri2)
        path = (rdflib.RDFS.subClassOf|rdflib.OWL.someValuesFrom)*rdflib.paths.ZeroOrMore
        eval_list = list(rdflib.paths.evalPath(self.__graph, (uriref1, path, uriref2)))
        return True if len(eval_list) > 0 else False

    def get_possible_ancestors(self, uri, parent_term, k=3) -> list:
        paths = [
            (rdflib.RDFS.subClassOf|rdflib.OWL.someValuesFrom)*rdflib.paths.ZeroOrMore,
            #  rdflib.RDFS.subClassOf*rdflib.paths.ZeroOrMore,
            ]
        ancestors =  [list(self.__graph.objects(rdflib.URIRef(uri), path)) for path in paths]
        ancestors = set([str(a) for anc in ancestors for a in anc if any([ns in str(a) for ns in namespaces.values()])])
        return self.__lookup.search_candidates(parent_term, k, list(ancestors))

    
    
    def save(self):
        pass

    def close():
        pass

#===============================================================================
