#P=/home/pawel/intel/oneapi/

comp="dnnl tbb compiler"

for item in $comp;
do

	P="/home/pawel/intel/oneapi/$item/latest/env/vars.sh"
        echo "$P"
        source $P
done


