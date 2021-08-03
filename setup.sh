dir=`pwd`
dir="$(basename $dir)"
if [ "$dir" != "grounding_consistent_VRD" ]; then
	echo -e "should run it from grounding_consistent_VRD dir"
elif [ "${BASH_SOURCE[0]}" != "${0}" ]; then
	conda create -n vrd python=3.8 -y
	eval $(conda shell.bash hook)
	conda activate vrd
	echo "installing requirements"
	pip install --user --requirement requirements.txt
else
	echo -e "remeber to source this script: . ./setup.sh"
fi
