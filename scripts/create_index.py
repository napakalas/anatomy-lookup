from anatomy_lookup import AnatomyLookup
import argparse
import logging

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Creating Anatomy Lookup index')
    parser.add_argument('--sckan-version', dest='sckan_version', default=None)
    args = parser.parse_args()
    
    lookup = AnatomyLookup()
    lookup.build_indexes(sckan_release = args.sckan_version)

    lookup.close()
