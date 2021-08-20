#!/bin/bash
#create folders to hold recipe output files

DIRS=./examples/V1/recipe_results/
echo "check $DIRS"

if [ -d "$DIRS" ];
then	
	if [ "$(ls -A $DIRS)" ];
	then
		rm ${DIRS}/*
		echo "remove files under $DIRS"
	fi
else
	mkdir ${DIRS}
	echo "create $DIRS"
fi

recipedirs="RecipeA RecipeB RecipeC RecipeD RecipeE RecipeF RecipeAlpha1 RecipeAlpha2 SimpleDemo"
for dir in $recipedirs; do
	recipedir=${DIRS}${dir}
	echo "check $recipedir"
	
	if [ -d "$recipedir" ];
	then
		:
	else
		mkdir ${recipedir} 
		echo "create $recipedir"
	fi
	if [ "$(ls -A $recipedir)" ];
	then 
		rm ${recipedir}/* 
		echo "remove files under $recipedir"
	fi
done

