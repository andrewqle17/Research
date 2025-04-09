# Example input:
# canyonlakehairprod.com.	172800	in	ns	ns13.wixdns.net.
# fatalsign.com.	86400	in	ds	21562 8 2 F2396C87340168468CE4B1CDA0629A096AF2ED926BAE1B347979C42EB6161B3D
# ...

with open('output_3m.txt') as f, open('domain_text_3m.txt', mode='w') as output:
    lines = f.readlines()
    for line in lines:
        output.write(line.split()[0][:-5])
        output.write('\n')