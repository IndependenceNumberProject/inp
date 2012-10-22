def ke_search():
        order = 2
        while True:
            try:
                gen = graphs.nauty_geng("-c {0}".format(order))
                counter = 0
                print order
                while True:
                    try:
                        g = INPGraph(gen.next())

                        b = g.bidouble()
                        mu = g.matching_number()
                        b_mu = b.matching_number()
                        ke = g.is_KE()

                        #if not ((ke and b_mu == 2 * mu) or ((not ke) and b_mu == 2 * mu + 1)):
                        #if (2 * mu > b_mu) or (b_mu > 2 * mu + 1):
                        if ke and b_mu != 2 * mu:
                            return g
                        
                        counter += 1

                    except StopIteration:
                        order += 1
                        break

            except KeyboardInterrupt:
                sys.stdout.flush()
                print "\nStopped."
                return None