for i in {1..100}
do
	../raxml-ng --msa ../fasta/test$i.fasta --model Dayhoff --prefix T$i --threads 2 --seed 616
	sed -i '' -e $'s/>//g' T$i.raxml.bestTree
done
