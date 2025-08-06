import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PruningChartsComponent } from './pruning-charts.component';

describe('PruningChartsComponent', () => {
  let component: PruningChartsComponent;
  let fixture: ComponentFixture<PruningChartsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PruningChartsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PruningChartsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
