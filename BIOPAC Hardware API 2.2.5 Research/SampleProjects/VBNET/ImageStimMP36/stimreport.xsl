<?xml version="1.0" encoding="utf-8" ?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
	<xsl:template match="/">
		<html>
			<body>
				<h1 align="center">Stimulus Report</h1>
				<p align="center">Click thumbmail images to see actual image or waveform</p>
				<table align="center" border="1">
					<tr bgcolor="blue">
						<th align="center">Score</th>
						<th align="center">Image</th>
						<th align="center">Waveform</th>
					</tr>
					<xsl:for-each select="StimulusReport/StimulusList/Stimulus">
						<xsl:sort select="Score" data-type="number" order="descending" />
						<tr>
							<td align="center">
								<h2>
									<xsl:value-of select="Score" />
								</h2>
							</td>
							<td align="center">
								<a href="{Location}">
									<img src="{Location}" alt="Score: {Score}" width="150" height="100" border="0" />
								</a>
							</td>
							<td>
								<a href="{UniqueID}.jpeg">
									<img src="{UniqueID}.jpeg" alt="Waveform" width="500" height="100" border="0" />
								</a>
							</td>
						</tr>
					</xsl:for-each>
				</table>
			</body>
		</html>
	</xsl:template>
</xsl:stylesheet>