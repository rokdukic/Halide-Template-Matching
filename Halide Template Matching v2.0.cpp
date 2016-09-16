// Halide Template Matching v2.0.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace Halide;
using namespace Halide::Tools;

void stackSolution(Image<float> source, Image<float> templ) {


	Var x, y, x_outer, y_outer, x_inner, y_inner, tile_index;


	Func limit;

	limit = BoundaryConditions::constant_exterior(source, 1.0f);


	RDom matchDom(0, templ.width(), 0, templ.height());
	//    Expr score = sum(matchDom, pow(templ(matchDom.x, matchDom.y) - limit(searchDom.x + matchDom.x, searchDom.y + matchDom.y), 2)) / (templ.width() * templ.height());
	Func score("score");
	score(x, y) = sum(matchDom, pow(templ(matchDom.x, matchDom.y) - limit(x + matchDom.x, y + matchDom.y), 2));
	
	/*score
		.tile(x, y, x_outer, y_outer, x_inner, y_inner, 64, 64)
		.fuse(x_outer, y_outer, tile_index)
		.vectorize(x_inner, 4)
		//       .unroll(y_inner)
		.parallel(tile_index)
		.compute_root()
		;*/

	score
		.vectorize(x, 16)
		.parallel(y)
		.compute_root();

	RDom searchDom(0, source.width() - templ.width(), 0, source.height() - templ.height());

	Tuple searchBest = argmin(searchDom, score(searchDom.x, searchDom.y), "argmin");

	Func best;
	best(_) = searchBest;

	LARGE_INTEGER ctr1{ 0 }, ctr2{ 0 }, freq{ 0 };
	QueryPerformanceFrequency(&freq);

	QueryPerformanceCounter(&ctr1);
	best.compile_jit();
	QueryPerformanceCounter(&ctr2);
	std::cout << "Template compile: " << ((ctr2.QuadPart - ctr1.QuadPart) *1.0 / freq.QuadPart) << "\n";

	QueryPerformanceCounter(&ctr1);
	Realization re = best.realize();
	QueryPerformanceCounter(&ctr2);
	std::cout << "Template Matching: " << ((ctr2.QuadPart - ctr1.QuadPart) *1.0 / freq.QuadPart) << "\n";


	Func draw("draw");

	Image<int> x_coordinate(re[0]);
	Image<int> y_coordinate(re[1]);
	Image<float> s(re[2]);

	int bestX = x_coordinate(0);
	int bestY = y_coordinate(0);

	draw(x, y) = select
	(((x == bestX || x == bestX + templ.width()) && y >= bestY && y <= bestY + templ.height())
		|| ((y == bestY || y == bestY + templ.height()) && x >= bestX && x <= bestX + templ.width())
		, 0.0f, likely(limit(x, y))
	);

	draw
		.tile(x, y, x_outer, y_outer, x_inner, y_inner, 64, 64)
		.fuse(x_outer, y_outer, tile_index)
		.vectorize(x_inner, 4)
		//        .unroll(y_inner)
		.parallel(tile_index);


	Image<float> drawTest;

	QueryPerformanceCounter(&ctr1);
	draw.compile_jit();
	QueryPerformanceCounter(&ctr2);
	std::cout << "Drawing compile: " << ((ctr2.QuadPart - ctr1.QuadPart) *1.0 / freq.QuadPart) << "\n";

	QueryPerformanceCounter(&ctr1);
	drawTest = draw.realize(source.width(), source.height());
	QueryPerformanceCounter(&ctr2);
	std::cout << "Drawing: " << ((ctr2.QuadPart - ctr1.QuadPart) *1.0 / freq.QuadPart) << "\n";

	//    drawTest = draw.realize(source.width(), source.height());

	//save_image(drawTest, "C:\\Users\\Admin\\Desktop\\templateMatchingOpenCV\\clip\\HalideResult.png");
}



int main()
{
	Image<float> source;
	Image<float> templ = load_image("C:\\Users\\rok10\\Pictures\\Template_matching\\template.png");

	std::string directoryPath = "C:\\Users\\rok10\\Pictures\\Template_matching\\images\\";

	std::vector<std::string> files;

	for (auto& dirEntry : std::experimental::filesystem::recursive_directory_iterator(directoryPath)) {
		files.push_back(dirEntry.path().filename().generic_string());
	}
#pragma omp parallel for shared(templ)
	for (int i = 0; i < files.size(); i++) {
		source = load_image(directoryPath + files.at(i));
		stackSolution(source, templ);
		//printf("%s\n", files.at(i));
	}
	

	//stackSolution(source, templ);

    return 0;
}

