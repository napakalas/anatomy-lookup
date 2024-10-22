from anatomy_lookup import AnatomyLookup, get_uriref, get_uri

if __name__ == '__main__':
    lookup = AnatomyLookup()
    print(lookup.search('Wall of larynx'))
    print(lookup.get_ancestor('UBERON:0001869'))
    print(lookup.search_candidates('prostate epithelium', uri_candidates=['UBERON:0000002', 'UBERON:0000079'], k=10000, force=True))
    lookup.close()
