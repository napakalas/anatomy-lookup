from anatomy_lookup import AnatomyLookup
from pprint import pp, pprint

if __name__ == '__main__':
    lookup = AnatomyLookup()
    print("search('Wall of larynx'):")
    pprint(lookup.search('Wall of larynx'))
    print()

    print("lookup.get_ancestor('UBERON:0001869'):")
    pprint(lookup.get_ancestor('UBERON:0001869'))
    print()

    print("search_candidates('prostate epithelium', uri_candidates=['UBERON:0000002', 'UBERON:0000079'], force=True):")
    pprint(lookup.search_candidates('prostate epithelium', uri_candidates=['UBERON:0000002', 'UBERON:0000079'], force=True))
    print()

    print("search_with_scope('prostate epithelium', scope=['male reproductive system']):")
    pprint(lookup.search_with_scope('prostate epithelium', scope=['male reproductive system']))

    lookup.close()
