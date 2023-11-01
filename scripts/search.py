from anatomy_lookup import AnatomyLookup

if __name__ == '__main__':
    lookup = AnatomyLookup()

    print(lookup.search('Wall of larynx'))

    lookup.close()
