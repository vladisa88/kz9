const buildSlideMarkup = (count) => {
	let slideMarkup = '';
  for (var i = 1; i <= count; i++) {
  	slideMarkup += '<slide><img src="https://picsum.photos/300/100/" style="width: 300px; max-width: 100%;"></slide>'
  }
  return slideMarkup;
};

export default buildSlideMarkup;
